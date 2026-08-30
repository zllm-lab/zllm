//! GLM-5.2 对公共 XML 工具协议的 tokenizer 绑定。

use crate::{
    backend::TokenFence,
    runtime::{
        generation_guard::TokenFenceProgram,
        tool::{XmlToolFence, XmlToolState, XmlToolTokens},
    },
};

pub const TOOL_CALL_OPEN_TOKEN: u32 = 154_843;
pub const TOOL_CALL_CLOSE_TOKEN: u32 = 154_844;
pub const ARG_KEY_OPEN_TOKEN: u32 = 154_847;
pub const ARG_KEY_CLOSE_TOKEN: u32 = 154_848;
pub const ARG_VALUE_OPEN_TOKEN: u32 = 154_849;
pub const ARG_VALUE_CLOSE_TOKEN: u32 = 154_850;
pub const TOOL_TOKENS: [u32; 6] = [TOOL_CALL_OPEN_TOKEN, TOOL_CALL_CLOSE_TOKEN, ARG_KEY_OPEN_TOKEN, ARG_KEY_CLOSE_TOKEN, ARG_VALUE_OPEN_TOKEN, ARG_VALUE_CLOSE_TOKEN];

const GLM_XML_TOKENS: XmlToolTokens =
    XmlToolTokens { call_open: TOOL_CALL_OPEN_TOKEN, call_close: TOOL_CALL_CLOSE_TOKEN, key_open: ARG_KEY_OPEN_TOKEN, key_close: ARG_KEY_CLOSE_TOKEN, value_open: ARG_VALUE_OPEN_TOKEN, value_close: ARG_VALUE_CLOSE_TOKEN };

#[derive(Clone)]
pub struct Glm52ToolFence(XmlToolFence);

impl Glm52ToolFence {
    pub fn new(tools_enabled: bool) -> Self {
        Self(XmlToolFence::new(tools_enabled, GLM_XML_TOKENS))
    }

    pub fn state(&self) -> XmlToolState {
        self.0.state()
    }
}

impl TokenFenceProgram for Glm52ToolFence {
    fn fence(&self) -> TokenFence {
        self.0.fence()
    }

    fn advance(&mut self, token: u32) {
        self.0.advance(token);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_blocks_orphan_tool_tokens() {
        assert_eq!(Glm52ToolFence::new(false).fence().excluded(), TOOL_TOKENS);
    }

    #[test]
    fn arg_value_close_is_only_open_inside_value() {
        let mut fence = Glm52ToolFence::new(true);
        assert!(fence.fence().excluded().contains(&ARG_VALUE_CLOSE_TOKEN));
        for token in [TOOL_CALL_OPEN_TOKEN, ARG_KEY_OPEN_TOKEN, ARG_KEY_CLOSE_TOKEN, ARG_VALUE_OPEN_TOKEN] {
            fence.advance(token);
        }
        assert!(!fence.fence().excluded().contains(&ARG_VALUE_CLOSE_TOKEN));
    }
}
