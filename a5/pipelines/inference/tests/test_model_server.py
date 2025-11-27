import pytest
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open, call
import re

class TestModelServer:
    """Test suite for extract_file_id function"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory structure for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Create test directory structure
            test_dir1 = tmp_path / "test_dir1"
            test_dir1.mkdir()
            (test_dir1 / "file1.txt").write_text("content1")
            (test_dir1 / "file2.txt").write_text("content2")
            
            # Create nested directory
            nested = test_dir1 / "nested"
            nested.mkdir()
            (nested / "file3.txt").write_text("content3")
            
            # Create second test directory
            test_dir2 = tmp_path / "test_dir2"
            test_dir2.mkdir()
            (test_dir2 / "file4.txt").write_text("content4")
            
            # Create standalone file
            standalone = tmp_path / "standalone.txt"
            standalone.write_text("standalone content")
            
            yield {
                'root': tmp_path,
                'dir1': test_dir1,
                'dir2': test_dir2,
                'standalone': standalone,
                'zip_path': tmp_path / "output.zip"
            }
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests module"""
        with patch('pipelines.inference.model_server.requests') as mock:
            yield mock
    
    @pytest.fixture
    def mock_extract_file_id(self):
        """Mock extract_file_id function"""
        with patch('pipelines.inference.model_server.extract_file_id') as mock:
            yield mock
    
    def test_extract_from_standard_share_link(self):
        """Test extraction from standard Google Drive share link"""
        from pipelines.inference.model_server import extract_file_id
        
        link1 = "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=sharing"
        link2 = "https://drive.google.com/open?id=abc123XYZ-_"
        link3 = "https://drive.google.com/d/xyz789ABC"
        
        assert extract_file_id(link1) == "1a2b3c4d5e6f7g8h9i0j"
        assert extract_file_id(link2) == "abc123XYZ-_"
        assert extract_file_id(link3) == "xyz789ABC"
    
    def test_invalid_link_returns_none(self):
        """Test that invalid links return None"""
        from pipelines.inference.model_server import extract_file_id
        
        invalid_links = [
            "https://www.google.com",
            "not a link",
            "https://drive.google.com/",
            "",
        ]
        
        for link in invalid_links:
            assert extract_file_id(link) is None
        
    def test_zip_with_other_paths(self, temp_dir):
        """Test zipping with additional standalone files"""
        from pipelines.inference.model_server import zip_directory
        
        zip_directory(
            directory_paths=[temp_dir['dir1'], temp_dir['dir2']],
            other_paths=[temp_dir['standalone']],
            zip_path=temp_dir['zip_path']
        )
        
        with zipfile.ZipFile(temp_dir['zip_path'], 'r') as zf:
            file_list = zf.namelist()
            assert any("test_dir1" in f for f in file_list)
            assert any("test_dir2" in f for f in file_list)
            assert len(file_list) == 5  # 3 from dir1 + 1 standalone
            assert len([f for f in file_list if "standalone.txt" in f]) == 1
        
    def test_zip_preserves_directory_structure(self, temp_dir):
        """Test that nested directory structure is preserved"""
        from pipelines.inference.model_server import zip_directory
        
        zip_directory(
            directory_paths=[temp_dir['dir1']],
            other_paths=[],
            zip_path=temp_dir['zip_path']
        )
        
        with zipfile.ZipFile(temp_dir['zip_path'], 'r') as zf:
            # Verify nested file has correct path
            assert "test_dir1/nested/file3.txt" in zf.namelist()
            
            # Verify content is correct
            content = zf.read("test_dir1/file1.txt").decode('utf-8')
            assert content == "content1"
    
    def test_zip_file_not_empty(self, temp_dir):
        """Test that zip file is not empty after creation"""
        from pipelines.inference.model_server import zip_directory
        
        zip_directory(
            directory_paths=[temp_dir['dir1']],
            other_paths=[],
            zip_path=temp_dir['zip_path']
        )
        
        # Check file size is greater than 0
        assert temp_dir['zip_path'].stat().st_size > 0
    
    def test_zip_raises_error_if_empty(self, temp_dir):
        """Test that error is raised if zip would be empty"""
        from pipelines.inference.model_server import zip_directory
        
        # Create empty directory
        empty_dir = temp_dir['root'] / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="Zip file has no contents"):
            zip_directory(
                directory_paths=[empty_dir],
                other_paths=[],
                zip_path=temp_dir['zip_path']
            )
    
    def test_zip_compression(self, temp_dir):
        """Test that ZIP_DEFLATED compression is used"""
        from pipelines.inference.model_server import zip_directory
        
        zip_directory(
            directory_paths=[temp_dir['dir1']],
            other_paths=[],
            zip_path=temp_dir['zip_path']
        )
        
        with zipfile.ZipFile(temp_dir['zip_path'], 'r') as zf:
            # Check that compression was used
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_DEFLATED
    
    def test_successful_download(self, mock_requests, mock_extract_file_id, tmp_path):
        """Test successful file download"""
        from pipelines.inference.model_server import download_from_google_drive
        
        # Setup mocks
        mock_extract_file_id.return_value = "test_file_id"
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1024"
        mock_response.cookies.items.return_value = []
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        
        mock_session.get.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "downloaded_file.mp4"
        
        result = download_from_google_drive(
            "https://drive.google.com/file/d/test_file_id/view",
            str(output_path)
        )
        
        assert result is True
        assert output_path.exists()
        assert output_path.read_bytes() == b"chunk1chunk2"
    
    def test_download_with_confirmation_cookie(self, mock_requests, mock_extract_file_id, tmp_path):
        """Test download with confirmation cookie for large files"""
        from pipelines.inference.model_server import download_from_google_drive
        
        mock_extract_file_id.return_value = "test_file_id"
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1024"
        mock_response.cookies.items.return_value = [("download_warning", "confirm_token")]
        mock_response.iter_content.return_value = [b"data"]
        
        mock_session.get.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "large_file.mp4"
        
        result = download_from_google_drive(
            "https://drive.google.com/file/d/test_file_id/view",
            str(output_path)
        )
        
        # Verify second request was made with confirmation
        assert mock_session.get.call_count == 2
        second_call_url = mock_session.get.call_args_list[1][0][0]
        assert "confirm=confirm_token" in second_call_url
        assert result is True
    
    def test_download_invalid_link(self, mock_extract_file_id, tmp_path):
        """Test download with invalid link"""
        from pipelines.inference.model_server import download_from_google_drive
        
        mock_extract_file_id.return_value = None
        
        output_path = tmp_path / "file.mp4"
        
        result = download_from_google_drive("https://invalid-link.com",str(output_path))
        
        assert result is False
        assert not output_path.exists()
    
    def test_download_failed_download(self, mock_requests, mock_extract_file_id, tmp_path):
        """Test download with invalid link"""
        from pipelines.inference.model_server import download_from_google_drive
        
        # Setup mocks
        mock_extract_file_id.return_value = "test_file_id"
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "1024"
        mock_response.cookies.items.return_value = []
        mock_response.iter_content.return_value = None # Induce error in download itself
        
        mock_session.get.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "downloaded_file.mp4"
        
        result = download_from_google_drive(
            "https://drive.google.com/file/d/test_file_id/view",
            str(output_path)
        )
        
        assert result is False
    
    def test_download_network_error(self, mock_requests, mock_extract_file_id, tmp_path):
        """Test download when network error occurs"""
        from pipelines.inference.model_server import download_from_google_drive
        
        mock_extract_file_id.return_value = "test_file_id"
        
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Network error")
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "file.mp4"
        
        result = download_from_google_drive(
            "https://drive.google.com/file/d/test_file_id/view",
            str(output_path)
        )
        
        assert result is False
    
    def test_download_progress_tracking(self, mock_requests, mock_extract_file_id, tmp_path, capsys):
        """Test that download progress is displayed"""
        from pipelines.inference.model_server import download_from_google_drive
        
        mock_extract_file_id.return_value = "test_file_id"
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "100"
        mock_response.cookies.items.return_value = []
        mock_response.iter_content.return_value = [b"a" * 50, b"b" * 50]
        
        mock_session.get.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "file.mp4"
        
        download_from_google_drive(
            "https://drive.google.com/file/d/test_file_id/view",
            str(output_path)
        )
        
        captured = capsys.readouterr()
        # Should show progress and success message
        assert "Downloaded successfully" in captured.out
    
    def test_download_uses_correct_url_format(self, mock_requests, mock_extract_file_id, tmp_path):
        """Test that correct Google Drive download URL is constructed"""
        from pipelines.inference.model_server import download_from_google_drive
        
        file_id = "abc123xyz"
        mock_extract_file_id.return_value = file_id
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.headers.get.return_value = "0"
        mock_response.cookies.items.return_value = []
        mock_response.iter_content.return_value = [b"data"]
        
        mock_session.get.return_value = mock_response
        mock_requests.Session.return_value = mock_session
        
        output_path = tmp_path / "file.mp4"
        
        download_from_google_drive(
            f"https://drive.google.com/file/d/{file_id}/view",
            str(output_path)
        )
        
        # Verify correct URL was used
        called_url = mock_session.get.call_args_list[0][0][0]
        assert called_url == f"https://drive.google.com/uc?export=download&id={file_id}"
        assert mock_session.get.call_args_list[0][1]["stream"] is True
        
        