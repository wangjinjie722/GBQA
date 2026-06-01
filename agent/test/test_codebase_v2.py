import pytest
from unittest.mock import MagicMock
from agent.src.codebase_types import UniversalCodebaseAdapter

@pytest.fixture
def mock_shell_client():
    client = MagicMock()
    # Mock shell.run and its async nature
    client.shell.run.return_value = MagicMock(stdout="test_file.py\nmain.py")
    client._run_async = lambda x: x
    return client

def test_sandbox_list_files(mock_shell_client):
    adapter = UniversalCodebaseAdapter(shell_client=mock_shell_client)
    files = adapter.list_files()
    
    assert len(files) == 2
    assert files[0].path == "test_file.py"
    # Verify the command used find
    mock_shell_client.shell.run.assert_any_call("find /sandbox/software -maxdepth 2 -not -path '*/.*'")

def test_sandbox_read_file(mock_shell_client):
    mock_shell_client.shell.run.return_value = MagicMock(stdout="print('hello')")
    adapter = UniversalCodebaseAdapter(shell_client=mock_shell_client)
    
    content = adapter.read_file("main.py")
    assert content == "print('hello')"
    mock_shell_client.shell.run.assert_called_with("cat /sandbox/software/main.py")

def test_sandbox_search_code(mock_shell_client):
    mock_shell_client.shell.run.return_value = MagicMock(stdout="/sandbox/software/main.py:10:def hello():")
    adapter = UniversalCodebaseAdapter(shell_client=mock_shell_client)
    
    matches = adapter.search_code("def hello")
    assert len(matches) == 1
    assert matches[0]["path"] == "main.py"
    assert matches[0]["line"] == "10"
    assert matches[0]["content"] == "def hello():"
