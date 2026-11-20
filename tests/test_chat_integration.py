"""Exercise the actual dialog function without starting the database dashboard."""
import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from dashboard.ai_provider import AIResponse, AISettings, AIUnavailableError


class Session(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class DialogTests(unittest.TestCase):
    def run_dialog(self, response=None, error=None):
        source = Path(__file__).parents[1] / 'dashboard' / 'app.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        dialog = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'show_chatbot_dialog')
        dialog.decorator_list = []
        st = MagicMock()
        st.secrets = {}
        st.session_state = Session()
        st.selectbox.return_value = 'AC123'
        st.columns.return_value = [MagicMock(), MagicMock()]
        st.button.return_value = False
        st.chat_input.return_value = 'How is this unit?'
        hr = MagicMock()
        hr.__len__.return_value = 1
        hr.to_json.return_value = 'fleet context'
        generate = MagicMock(return_value=response, side_effect=error)
        namespace = dict(st=st, hr=hr, DEVICE_NAMES={}, engine=object(),
                         AISettings=AISettings, AIUnavailableError=AIUnavailableError,
                         generate_response=generate,
                         get_ac_historical_profile=MagicMock(return_value='history context'),
                         _build_system_prompt=MagicMock(return_value='complete prompt'))
        exec(compile(ast.Module(body=[dialog], type_ignores=[]), str(source), 'exec'), namespace)
        namespace['show_chatbot_dialog']()
        return st, generate

    def test_local_chat_works_without_gemini_secret(self):
        st, generate = self.run_dialog(AIResponse('local answer', 'Ollama', 'llama3.2:3b'))
        self.assertEqual(generate.call_args.args[0], 'complete prompt')
        self.assertEqual(st.session_state.chat_history[-1]['content'], 'local answer')
        self.assertEqual(st.session_state.chat_history[-1]['provider'], 'Ollama')
        st.error.assert_not_called()

    def test_fallback_metadata_is_retained(self):
        st, _ = self.run_dialog(AIResponse('cloud answer', 'Gemini', 'custom', 'request timed out'))
        message = st.session_state.chat_history[-1]
        self.assertEqual(message['provider'], 'Gemini')
        self.assertEqual(message['fallback_reason'], 'request timed out')

    def test_outage_keeps_user_question_and_displays_error(self):
        st, _ = self.run_dialog(error=AIUnavailableError('Both services unavailable'))
        self.assertEqual(len(st.session_state.chat_history), 2)
        self.assertEqual(st.session_state.chat_history[0]['content'], 'How is this unit?')
        self.assertIn('Both services unavailable', st.session_state.chat_history[-1]['content'])


if __name__ == '__main__':
    unittest.main()
