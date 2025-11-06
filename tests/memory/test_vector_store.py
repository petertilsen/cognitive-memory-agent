"""Unit tests for vector store functionality."""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.memory.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_embedding_model = Mock()
        self.mock_embedding_model.config = {"model_id": "test-model"}
        
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_vector_store_initialization(self, mock_client_class):
        """Test VectorStore initialization."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(
            embedding_model=self.mock_embedding_model,
            chroma_host="test_host",
            chroma_port=9000,
            collection_name="test_collection"
        )
        
        self.assertEqual(store.collection_name, "test_collection")
        self.assertEqual(store.embedding_dim, 1536)
        mock_client_class.assert_called_once()
        mock_client.get_or_create_collection.assert_called_once()
        
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_embed_method(self, mock_client_class):
        """Test embedding generation."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock the Bedrock model response
        mock_response = {'body': Mock()}
        mock_response['body'].read.return_value = '{"embedding": [0.1, 0.2, 0.3]}'
        self.mock_embedding_model.client.invoke_model.return_value = mock_response
        
        store = VectorStore(self.mock_embedding_model)
        embedding = store.embed("test text")
        
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
            
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_add_document(self, mock_client_class):
        """Test adding a document to the vector store."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(self.mock_embedding_model)
        
        # Mock embedding generation
        with patch.object(store, 'embed', return_value=[0.1, 0.2, 0.3]):
            doc_id = store.add("test content", {"source": "test"})
            
            self.assertIsInstance(doc_id, str)
            mock_collection.add.assert_called_once()
            
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_search_similar(self, mock_client_class):
        """Test similarity search."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Mock search results
        mock_collection.query.return_value = {
            'ids': [['id1', 'id2']],
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'source': 'test1'}, {'source': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        store = VectorStore(self.mock_embedding_model)
        
        with patch.object(store, 'embed', return_value=[0.1, 0.2, 0.3]):
            results = store.search("test query", top_k=2)
            
            self.assertEqual(len(results), 2)
            # Results are tuples: (doc_id, similarity, document, metadata)
            self.assertEqual(results[0][2], 'doc1')  # document
            self.assertEqual(results[0][3]['source'], 'test1')  # metadata
            
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_count_documents(self, mock_client_class):
        """Test document count."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.return_value = 5
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(self.mock_embedding_model)
        count = store.count()
        
        self.assertEqual(count, 5)
        mock_collection.count.assert_called_once()
        
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_reset_collection(self, mock_client_class):
        """Test collection reset."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(self.mock_embedding_model)
        store.reset()
        
        mock_client.delete_collection.assert_called_once()
        
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_error_handling_in_embed(self, mock_client_class):
        """Test error handling in embed method."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(self.mock_embedding_model)
        
        with patch('requests.post', side_effect=Exception("Network error")):
            embedding = store.embed("test text")
            
            # Should return zero vector on error
            self.assertEqual(len(embedding), store.embedding_dim)
            self.assertTrue(all(x == 0.0 for x in embedding))
            
    @patch('src.memory.vector_store.chromadb.HttpClient')
    def test_error_handling_in_count(self, mock_client_class):
        """Test error handling in count method."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.count.side_effect = Exception("Database error")
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        store = VectorStore(self.mock_embedding_model)
        count = store.count()
        
        # Should return 0 on error
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()
