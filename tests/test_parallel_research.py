import argparse
import asyncio
import json

from unittest.mock import patch

import pytest

pytest.importorskip('mcp')  # Only this optional example needs the MCP SDK.

import httpx

from agentron.tool.manager import CoreToolManager
from examples import parallel_research as example


class MCPFixture:
    """Exercise the real MCP client while replacing only its HTTP boundary."""

    def __init__(self, mode='success'):
        self.mode = mode
        self.requests = []
        self.clients = []
        self.called = asyncio.Event()

    async def __call__(self, request):
        message = json.loads(request.content)
        self.requests.append((request, message))
        assert str(request.url) == example.MCP_URL
        assert request.headers['User-Agent'] == example.USER_AGENT
        assert 'authorization' not in request.headers
        if 'id' not in message:
            return httpx.Response(202)
        method = message['method']
        if method == 'initialize':
            result = {
                'protocolVersion': message['params']['protocolVersion'],
                'capabilities': {'tools': {}},
                'serverInfo': {'name': 'fixture', 'version': '1.0'},
            }
        elif method == 'tools/list':
            result = {'tools': []}
        elif method == 'tools/call':
            self.called.set()
            if self.mode == 'timeout':
                await asyncio.Event().wait()
            if self.mode == 'protocol_error':
                return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'error': {'code': -32602, 'message': 'Invalid search arguments'}})
            name = message['params']['name']
            text = 'Python release notes: https://docs.python.org/3/whatsnew/3.14.html' if name == 'web_search' else 'Python 3.14 release highlights from the source page.'
            result = {'content': [{'type': 'text', 'text': text}]}
            if self.mode == 'tool_error':
                result = {'isError': True, 'content': [{'type': 'text', 'text': 'Usage limit reached'}]}
            elif self.mode == 'empty':
                result = {'content': []}
        else:
            raise AssertionError(f'Unexpected method: {method}')
        return httpx.Response(200, json={'jsonrpc': '2.0', 'id': message['id'], 'result': result})

    def install(self):
        real_client = httpx.AsyncClient

        def client(**kwargs):
            instance = real_client(**kwargs, transport=httpx.MockTransport(self))
            self.clients.append(instance)
            return instance

        return patch.object(example.httpx, 'AsyncClient', side_effect=client)


def tool_call(name='web_search', arguments=None):
    return {'type': 'tool_call', 'id': 'call-1', 'name': name, 'arguments': arguments or {'objective': 'Python release highlights', 'search_queries': ['Python release notes']}}


def test_registered_tools_serialize_requests_and_return_sources():
    async def run():
        fixture = MCPFixture()
        with fixture.install():
            async with example.connect_parallel() as tools:
                manager = CoreToolManager([tools.web_search, tools.web_fetch])
                assert [schema['name'] for schema in manager.schema] == ['web_search', 'web_fetch']
                assert manager.schema[0]['parameters']['required'] == ['objective', 'search_queries']
                assert manager.schema[1]['parameters']['required'] == ['url']
                await tools.session.list_tools()
                search = await manager(tool_call())
                fetch = await manager(tool_call('web_fetch', {'url': 'https://docs.python.org/3/whatsnew/3.14.html'}))
                assert search['success'] and 'https://docs.python.org/' in search['content']['text']
                assert fetch['success'] and 'source page' in fetch['content']['text']
        calls = [message['params'] for _, message in fixture.requests if message['method'] == 'tools/call']
        assert calls[0]['arguments'] == {
            'objective': 'Python release highlights',
            'search_queries': ['Python release notes'],
            'session_id': tools.session_id,
        }
        assert calls[1]['arguments'] == {'urls': ['https://docs.python.org/3/whatsnew/3.14.html'], 'session_id': tools.session_id}
        assert len(tools.session_id) == 32

    asyncio.run(run())


@pytest.mark.parametrize('mode, expected', [('tool_error', 'Usage limit reached'), ('empty', 'no text content'), ('protocol_error', 'Invalid search arguments'), ('timeout', 'Timed out')])
def test_failures_are_returned_as_failed_tool_results(mode, expected):
    async def run():
        fixture = MCPFixture(mode)
        with fixture.install():
            async with example.connect_parallel(timeout=0.1) as tools:
                result = await CoreToolManager([tools.web_search])(tool_call())
                assert not result['success']
                assert expected in result['content']['text']
                assert fixture.called.is_set()

    asyncio.run(run())


def test_cancellation_propagates_and_closes_connection():
    async def run():
        fixture = MCPFixture('timeout')
        with fixture.install():
            async with example.connect_parallel() as tools:
                task = asyncio.create_task(CoreToolManager([tools.web_search])(tool_call()))
                await asyncio.wait_for(fixture.called.wait(), timeout=1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert all(client.is_closed for client in fixture.clients)

    asyncio.run(run())


def test_documented_agent_path_consumes_search_and_fetch(capsys):
    async def run():
        fixture = MCPFixture()
        model = {
            'id': 'chosen-model', 'name': 'Fixture', 'api': 'openai-completions', 'provider': 'fixture',
            'base_url': 'https://model.example', 'reasoning': False, 'input': ['text'],
            'cost': {'input': 0, 'output': 0, 'cache_read': 0, 'cache_write': 0},
            'context_window': 32000, 'max_tokens': 4096,
        }
        turns = 0

        async def backend(self, *, messages, **kwargs):
            nonlocal turns
            assert self.model == model
            assert self.api_key == 'fixture-model-key'
            if turns == 0:
                content = [tool_call()]
            elif turns == 1:
                assert messages[-1]['result']['success']
                assert 'https://docs.python.org/' in messages[-1]['result']['content']['text']
                content = [tool_call('web_fetch', {'url': 'https://docs.python.org/3/whatsnew/3.14.html'})]
            else:
                assert messages[-1]['result']['success']
                assert 'source page' in messages[-1]['result']['content']['text']
                content = [{'type': 'text', 'text': 'Research answer with source citation.'}]
            turns += 1
            return {'mtype': 'assistant', 'id': str(turns), 'timestamp': 0, 'content': content}

        with (
            fixture.install(),
            patch('agentron.utils.agent.get_model', return_value=model) as get_model,
            patch('agentron.utils.agent.resolve_api_key', return_value='fixture-model-key'),
            patch('agentron.rpc.flux.FluxBackend.__call__', backend),
            patch('agentron.utils.agent.TerminalOutput'),
        ):
            await example.main(argparse.Namespace(search=None, fetch=None, prompt='Research Python releases', model='fixture:chosen-model'))
            get_model.assert_called_once_with('fixture:chosen-model')
        assert turns == 3

    asyncio.run(run())
    assert 'Research answer with source citation.' in capsys.readouterr().out


@pytest.mark.parametrize('mode', ['search', 'fetch'])
def test_direct_commands_do_not_construct_a_model(mode, capsys):
    async def run():
        fixture = MCPFixture()
        args = argparse.Namespace(search='Python release highlights' if mode == 'search' else None, fetch='https://docs.python.org/' if mode == 'fetch' else None)
        with fixture.install(), patch.object(example, 'make_agent', side_effect=AssertionError('No model should be used')):
            await example.main(args)

    asyncio.run(run())
    assert 'Python' in capsys.readouterr().out
