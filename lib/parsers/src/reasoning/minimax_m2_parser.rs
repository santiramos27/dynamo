// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{ParserResult, ReasoningParser};

const THINK_START_TOKEN: &str = "<think>";
const THINK_END_TOKEN: &str = "</think>";

/// Returns the length of the longest suffix of `s` that is also a prefix of
/// `delim`.
fn overlap(s: &str, delim: &str) -> usize {
    let max = delim.len().min(s.len());
    for i in (1..=max).rev() {
        if !delim.is_char_boundary(i) {
            continue;
        }
        if s.ends_with(&delim[..i]) {
            return i;
        }
    }
    0
}

/// MiniMax M2 reasoning parser.
///
/// MiniMax M2 normally omits the `<think>` opener in generated output. The
/// stream starts in reasoning mode, the first `</think>` closes reasoning, and
/// all subsequent bytes are normal content for the downstream tool-call parser.
///
/// This matches vLLM's `MiniMaxM2ReasoningParser` contract while using text
/// boundary buffering because Dynamo's native parser interface receives text
/// deltas rather than tokenizer ids. Dynamo also strips an unexpected leading
/// `<think>` opener so the marker does not leak if a backend emits one anyway.
#[derive(Debug)]
pub struct MiniMaxM2ReasoningParser {
    in_reasoning: bool,
    buffer: String,
    checked_optional_start: bool,
}

impl MiniMaxM2ReasoningParser {
    pub fn new() -> Self {
        Self {
            in_reasoning: true,
            buffer: String::new(),
            checked_optional_start: false,
        }
    }

    fn strip_optional_start_for_batch(text: &str) -> &str {
        text.strip_prefix(THINK_START_TOKEN).unwrap_or(text)
    }

    fn strip_optional_start_for_stream(&mut self) -> bool {
        if self.checked_optional_start {
            return false;
        }

        if self.buffer.starts_with(THINK_START_TOKEN) {
            self.buffer.drain(..THINK_START_TOKEN.len());
            self.checked_optional_start = true;
            return true;
        }

        if THINK_START_TOKEN.starts_with(self.buffer.as_str())
            && self.buffer.len() < THINK_START_TOKEN.len()
        {
            return false;
        }

        self.checked_optional_start = true;
        false
    }
}

impl Default for MiniMaxM2ReasoningParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningParser for MiniMaxM2ReasoningParser {
    fn set_in_reasoning(&mut self, in_reasoning: bool) {
        self.in_reasoning = in_reasoning;
        if in_reasoning {
            self.checked_optional_start = false;
        }
    }

    fn detect_and_parse_reasoning(&mut self, text: &str, _token_ids: &[u32]) -> ParserResult {
        let text = Self::strip_optional_start_for_batch(text);
        if let Some(end_idx) = text.find(THINK_END_TOKEN) {
            ParserResult {
                reasoning_text: text[..end_idx].to_string(),
                normal_text: text[end_idx + THINK_END_TOKEN.len()..].to_string(),
            }
        } else {
            ParserResult {
                reasoning_text: text.to_string(),
                normal_text: String::new(),
            }
        }
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        _token_ids: &[u32],
    ) -> ParserResult {
        if !self.in_reasoning {
            return ParserResult {
                normal_text: text.to_string(),
                reasoning_text: String::new(),
            };
        }

        self.buffer.push_str(text);

        loop {
            if self.strip_optional_start_for_stream() {
                continue;
            }
            break;
        }

        if !self.checked_optional_start {
            return ParserResult::default();
        }

        if let Some(end_idx) = self.buffer.find(THINK_END_TOKEN) {
            let reasoning_text = self.buffer[..end_idx].to_string();
            let normal_start = end_idx + THINK_END_TOKEN.len();
            let normal_text = self.buffer[normal_start..].to_string();
            self.buffer.clear();
            self.in_reasoning = false;
            return ParserResult {
                normal_text,
                reasoning_text,
            };
        }

        // Hold a split closing token like `</th` across chunks. A lone `<`
        // is emitted immediately so incomplete reasoning text is not lost if
        // the stream ends without ever producing `</think>`.
        let overlap_len = overlap(&self.buffer, THINK_END_TOKEN);
        if overlap_len >= 2 {
            let safe_end = self.buffer.len() - overlap_len;
            let reasoning_text = self.buffer[..safe_end].to_string();
            self.buffer = self.buffer[safe_end..].to_string();
            ParserResult {
                normal_text: String::new(),
                reasoning_text,
            }
        } else {
            let reasoning_text = std::mem::take(&mut self.buffer);
            ParserResult {
                normal_text: String::new(),
                reasoning_text,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_and_parse_splits_at_end_token() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let result = parser.detect_and_parse_reasoning("reasoning</think>answer", &[]);
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_detect_and_parse_no_end_token_is_all_reasoning() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let result = parser.detect_and_parse_reasoning("reasoning only", &[]);
        assert_eq!(result.reasoning_text, "reasoning only");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_detect_and_parse_strips_unexpected_start_token() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let result = parser.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(result.reasoning_text, "reasoning");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_streaming_splits_end_token_across_chunks() {
        let mut parser = MiniMaxM2ReasoningParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("I need ", &[]);
        assert_eq!(r1.reasoning_text, "I need ");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("to think</th", &[]);
        assert_eq!(r2.reasoning_text, "to think");
        assert_eq!(r2.normal_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("ink>Answer", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "Answer");
    }

    #[test]
    fn test_streaming_exact_end_token_transitions_to_content() {
        let mut parser = MiniMaxM2ReasoningParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("reasoning", &[]);
        assert_eq!(r1.reasoning_text, "reasoning");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("</think>", &[]);
        assert_eq!(r2.reasoning_text, "");
        assert_eq!(r2.normal_text, "");

        let r3 = parser.parse_reasoning_streaming_incremental("Answer", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "Answer");
    }

    #[test]
    fn test_streaming_tool_call_after_reasoning_is_normal_text() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let result = parser.parse_reasoning_streaming_incremental(
            "thinking</think><minimax:tool_call><invoke name=\"get_weather\">",
            &[],
        );
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(
            result.normal_text,
            "<minimax:tool_call><invoke name=\"get_weather\">"
        );
    }

    #[test]
    fn test_streaming_strips_unexpected_start_token_across_chunks() {
        let mut parser = MiniMaxM2ReasoningParser::new();

        let r1 = parser.parse_reasoning_streaming_incremental("<thi", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        let r2 = parser.parse_reasoning_streaming_incremental("nk>thinking</think>Answer", &[]);
        assert_eq!(r2.reasoning_text, "thinking");
        assert_eq!(r2.normal_text, "Answer");
    }

    #[test]
    fn test_streaming_no_end_token_is_reasoning() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let r1 = parser.parse_reasoning_streaming_incremental("reason", &[]);
        let r2 = parser.parse_reasoning_streaming_incremental("ing", &[]);
        assert_eq!(
            format!("{}{}", r1.reasoning_text, r2.reasoning_text),
            "reasoning"
        );
        assert_eq!(r1.normal_text, "");
        assert_eq!(r2.normal_text, "");
    }

    #[test]
    fn test_streaming_lone_angle_bracket_is_not_buffered_forever() {
        let mut parser = MiniMaxM2ReasoningParser::new();
        let result = parser.parse_reasoning_streaming_incremental("reasoning <", &[]);
        assert_eq!(result.reasoning_text, "reasoning <");
        assert_eq!(result.normal_text, "");
    }
}
