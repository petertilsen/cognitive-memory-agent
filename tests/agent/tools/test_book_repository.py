"""Unit tests for book repository tools."""

import unittest
import sys
import os
from unittest.mock import Mock, patch


from src.agent.tools.book_repository import (
    search_gutenberg_books,
    fetch_book_content,
    search_openlibrary_books
)


class TestBookRepositoryTools(unittest.TestCase):
    """Test cases for book repository tools."""
    
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_gutenberg_books_success(self, mock_get):
        """Test successful Gutenberg book search."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "id": 1,
                    "title": "Test Book",
                    "authors": [{"name": "Test Author"}],
                    "subjects": ["Fiction"],
                    "download_count": 100,
                    "formats": {
                        "text/plain; charset=utf-8": "http://example.com/book.txt",
                        "text/html": "http://example.com/book.html"
                    }
                }
            ],
            "count": 1
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_gutenberg_books("test query", max_results=5)
        
        self.assertIn("Test Book", result)
        self.assertIn("Test Author", result)
        self.assertIn("success", result)
        mock_get.assert_called_once()
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_gutenberg_books_error(self, mock_get):
        """Test Gutenberg book search with error."""
        mock_get.side_effect = Exception("Network error")
        
        result = search_gutenberg_books("test query")
        
        self.assertIn("error", result)
        self.assertIn("Network error", result)
        self.assertIn("success", result)
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_fetch_book_content_success(self, mock_get):
        """Test successful book content fetching."""
        mock_response = Mock()
        mock_response.text = "This is the book content. " * 100  # Long content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = fetch_book_content("http://example.com/book.txt", max_chars=500)
        
        self.assertIn("This is the book content", result)
        self.assertLessEqual(len(result), 500 + 50)  # Account for truncation message
        mock_get.assert_called_once_with("http://example.com/book.txt", timeout=15)
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_fetch_book_content_truncation(self, mock_get):
        """Test book content truncation for long content."""
        long_content = "A" * 20000
        mock_response = Mock()
        mock_response.text = long_content
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = fetch_book_content("http://example.com/book.txt", max_chars=1000)
        
        self.assertIn("[Content truncated for memory efficiency...]", result)
        self.assertLessEqual(len(result), 1100)  # Should be around max_chars + truncation message
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_fetch_book_content_error(self, mock_get):
        """Test book content fetching with error."""
        mock_get.side_effect = Exception("404 Not Found")
        
        result = fetch_book_content("http://example.com/nonexistent.txt")
        
        self.assertIn("Error fetching book content", result)
        self.assertIn("404 Not Found", result)
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_openlibrary_books_success(self, mock_get):
        """Test successful Open Library book search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "docs": [
                {
                    "key": "/works/OL123W",
                    "title": "Machine Learning Book",
                    "author_name": ["John Doe", "Jane Smith"],
                    "first_publish_year": 2020,
                    "subject": ["Computer Science", "AI", "Machine Learning"],
                    "ia": ["machinelearning001"]
                },
                {
                    "key": "/works/OL456W",
                    "title": "AI Fundamentals",
                    "author_name": ["Alice Johnson"],
                    "first_publish_year": 2019,
                    "subject": ["Artificial Intelligence"],
                    "ia": []  # No Internet Archive access
                }
            ],
            "numFound": 2
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_openlibrary_books("machine learning", max_results=5)
        
        self.assertIn("Machine Learning Book", result)
        self.assertIn("John Doe", result)
        self.assertIn("AI Fundamentals", result)
        self.assertIn("success", result)
        self.assertIn("archive.org/stream/machinelearning001", result)
        mock_get.assert_called_once()
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_openlibrary_books_with_ia_access(self, mock_get):
        """Test Open Library search with Internet Archive access."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "docs": [
                {
                    "key": "/works/OL123W",
                    "title": "Test Book",
                    "author_name": ["Test Author"],
                    "ia": ["testbook001", "testbook002"]  # Multiple IA IDs
                }
            ],
            "numFound": 1
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_openlibrary_books("test")
        
        # Should use first IA ID
        self.assertIn("archive.org/stream/testbook001", result)
        self.assertNotIn("testbook002", result)
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_openlibrary_books_no_ia_access(self, mock_get):
        """Test Open Library search without Internet Archive access."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "docs": [
                {
                    "key": "/works/OL123W",
                    "title": "Test Book",
                    "author_name": ["Test Author"],
                    "ia": None  # No IA access
                }
            ],
            "numFound": 1
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_openlibrary_books("test")
        
        self.assertIn("Test Book", result)
        self.assertIn("ia_url", result)
        # Should have null IA URL
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_openlibrary_books_error(self, mock_get):
        """Test Open Library search with error."""
        mock_get.side_effect = Exception("API error")
        
        result = search_openlibrary_books("test query")
        
        self.assertIn("error", result)
        self.assertIn("API error", result)
        self.assertIn("success", result)
        
    @patch('src.agent.tools.book_repository.requests.get')
    def test_search_openlibrary_books_empty_results(self, mock_get):
        """Test Open Library search with no results."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "docs": [],
            "numFound": 0
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = search_openlibrary_books("nonexistent topic")
        
        self.assertIn("success", result)
        self.assertIn("total_found", result)
        # Should handle empty results gracefully
        
    def test_search_gutenberg_books_default_parameters(self):
        """Test Gutenberg search with default parameters."""
        with patch('src.agent.tools.book_repository.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"results": [], "count": 0}
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            search_gutenberg_books("test")
            
            # Should use default max_results=5
            call_args = mock_get.call_args
            self.assertEqual(call_args[1]['params']['page_size'], 5)
            
    def test_fetch_book_content_default_parameters(self):
        """Test book content fetching with default parameters."""
        with patch('src.agent.tools.book_repository.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = "Short content"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            result = fetch_book_content("http://example.com/book.txt")
            
            # Should use default max_chars=10000
            self.assertEqual(result, "Short content")
            # Should not be truncated
            self.assertNotIn("truncated", result)


if __name__ == '__main__':
    unittest.main()
