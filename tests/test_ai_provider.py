import os
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

import requests

from dashboard.ai_provider import AISettings, AIUnavailableError, generate_response


def reply(data, status=200):
    response = MagicMock()
    response.__enter__.return_value = response
    response.json.return_value = data
    response.status_code = status
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    return response


def gemini_reply():
    return reply({'candidates': [{'finishReason': 'STOP', 'content': {
        'parts': [{'text': 'Cloud answer'}]}}]})


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.settings = AISettings(gemini_api_key='test-key')
        self.patcher = patch('dashboard.ai_provider.requests.post')
        self.post = self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_local_success_never_calls_gemini_and_needs_no_key(self):
        self.post.return_value = reply({'response': ' Local answer ', 'done': True})
        result = generate_response('fleet question', AISettings())
        self.assertEqual((result.provider, result.text), ('Ollama', 'Local answer'))
        self.post.assert_called_once()
        args, kwargs = self.post.call_args
        self.assertEqual(args[0], 'http://localhost:11434/api/generate')
        self.assertFalse(kwargs['json']['stream'])
        self.assertEqual(kwargs['timeout'], (3.0, 180.0))

    def test_failures_fall_back_with_same_prompt(self):
        failures = [
            requests.ConnectionError('Docker stopped'),
            requests.Timeout('model stalled'),
            reply({}, 404), reply({}, 500),
            reply({'response': '   ', 'done': True}),
            reply({'response': 'partial', 'done': False}),
            reply({'error': 'unavailable'}), reply([]),
        ]
        invalid_json = reply({})
        invalid_json.json.side_effect = ValueError('bad JSON')
        failures.append(invalid_json)
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.post.reset_mock()
                self.post.side_effect = [failure, gemini_reply()]
                result = generate_response('same context', self.settings)
                self.assertEqual(result.provider, 'Gemini')
                self.assertTrue(result.fallback_reason)
                self.assertEqual(self.post.call_count, 2)
                args, kwargs = self.post.call_args
                self.assertIn('gemini-3.5-flash:generateContent', args[0])
                self.assertNotIn('test-key', args[0])
                self.assertEqual(kwargs['headers']['x-goog-api-key'], 'test-key')
                self.assertEqual(kwargs['json']['contents'][0]['parts'][0]['text'], 'same context')
                self.assertEqual(kwargs['timeout'], (3.0, 45.0))

    def test_recovers_to_local_on_next_request(self):
        self.post.side_effect = [requests.ConnectionError(), gemini_reply(),
                                 reply({'response': 'Recovered', 'done': True})]
        self.assertEqual(generate_response('first', self.settings).provider, 'Gemini')
        self.assertEqual(generate_response('second', self.settings).provider, 'Ollama')
        self.assertEqual(self.post.call_count, 3)

    def test_missing_key_fails_gracefully_without_cloud_call(self):
        self.post.side_effect = requests.ConnectionError()
        with self.assertRaisesRegex(AIUnavailableError, 'GEMINI_API_KEY is not configured'):
            generate_response('question', AISettings())
        self.post.assert_called_once()

    def test_both_fail_without_exposing_raw_errors(self):
        self.post.side_effect = [requests.ConnectionError('secret prompt'),
                                 requests.ConnectionError('test-key')]
        with self.assertRaises(AIUnavailableError) as caught:
            generate_response('question', self.settings)
        self.assertNotIn('test-key', str(caught.exception))
        self.assertNotIn('secret prompt', str(caught.exception))
        self.assertIn('Ollama:', str(caught.exception))
        self.assertIn('Gemini fallback:', str(caught.exception))

    def test_gemini_errors_and_empty_or_blocked_responses(self):
        for cloud in [reply({}, 401), reply({}, 429), reply({}, 503),
                      requests.Timeout(), reply({}), reply({'candidates': []}),
                      reply({'candidates': [{'finishReason': 'SAFETY', 'content': {
                          'parts': [{'text': 'blocked'}]}}]})]:
            with self.subTest(cloud=type(cloud).__name__):
                self.post.side_effect = [requests.ConnectionError(), cloud]
                with self.assertRaises(AIUnavailableError):
                    generate_response('question', self.settings)

    def test_thinking_is_not_shown(self):
        self.post.side_effect = [requests.ConnectionError(), reply({'candidates': [{
            'content': {'parts': [{'thought': True, 'text': 'internal'},
                                  {'text': 'final'}]}}]})]
        self.assertEqual(generate_response('question', self.settings).text, 'final')

    def test_invalid_ollama_configuration_can_fall_back(self):
        for url in ['bad', 'http://[malformed']:
            with self.subTest(url=url):
                self.post.reset_mock()
                self.post.return_value = gemini_reply()
                result = generate_response('question', replace(self.settings, ollama_base_url=url))
                self.assertEqual(result.provider, 'Gemini')
                self.post.assert_called_once()


class SettingsTests(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_and_secret_settings(self):
        settings = AISettings.from_mapping({'OLLAMA_MODEL': 'custom:7b',
            'OLLAMA_BASE_URL': 'http://ollama:11434/', 'GEMINI_MODEL': 'models/custom',
            'GEMINI_API_KEY': 'hidden'})
        self.assertEqual(settings.ollama_model, 'custom:7b')
        self.assertEqual(settings.ollama_base_url, 'http://ollama:11434')
        self.assertEqual(settings.gemini_model, 'custom')
        self.assertNotIn('hidden', repr(settings))

    @patch.dict(os.environ, {'OLLAMA_MODEL': 'environment:3b'}, clear=True)
    def test_environment_overrides_secrets(self):
        self.assertEqual(AISettings.from_mapping({'OLLAMA_MODEL': 'secret:7b'}).ollama_model,
                         'environment:3b')

    @patch.dict(os.environ, {}, clear=True)
    def test_rejects_invalid_timeouts(self):
        for value in ['0', '-1', 'nan', 'inf', 'abc']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive'):
                AISettings.from_mapping({'OLLAMA_TIMEOUT_SECONDS': value})


if __name__ == '__main__':
    unittest.main()
