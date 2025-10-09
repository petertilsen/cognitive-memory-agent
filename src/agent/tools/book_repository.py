"""Online book repository tool for accessing free books and texts."""

import requests
from typing import List, Dict, Optional
from strands import tool

from ....config.settings import get_logger

logger = get_logger("agent.tools.book_repository")


@tool
def search_gutenberg_books(query: str, max_results: int = 5) -> str:
    """Search Project Gutenberg for free books related to a topic.
    
    Args:
        query: Search term (e.g., "machine learning", "philosophy", "history")
        max_results: Maximum number of books to return
    
    Returns:
        JSON string with book information including titles, authors, and download URLs
    """
    logger.debug(f"Searching Project Gutenberg for: {query}")
    
    try:
        # Project Gutenberg API search
        url = "https://gutendex.com/books/"
        params = {
            "search": query,
            "page_size": max_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        books = []
        
        for book in data.get("results", []):
            book_info = {
                "id": book.get("id"),
                "title": book.get("title", "Unknown Title"),
                "authors": [author.get("name", "Unknown") for author in book.get("authors", [])],
                "subjects": book.get("subjects", []),
                "download_count": book.get("download_count", 0),
                "formats": book.get("formats", {}),
                "text_url": book.get("formats", {}).get("text/plain; charset=utf-8"),
                "html_url": book.get("formats", {}).get("text/html")
            }
            books.append(book_info)
        
        result = {
            "query": query,
            "total_found": data.get("count", 0),
            "books": books,
            "success": True
        }
        
        logger.info(f"Found {len(books)} books for query: {query}")
        return str(result)
        
    except Exception as e:
        logger.error(f"Failed to search Gutenberg books: {e}")
        return str({"error": str(e), "success": False})


@tool
def fetch_book_content(book_url: str, max_chars: int = 10000) -> str:
    """Fetch content from a book URL (Project Gutenberg text).
    
    Args:
        book_url: URL to the book's text content
        max_chars: Maximum characters to fetch (for memory efficiency)
    
    Returns:
        Book content as text string
    """
    logger.debug(f"Fetching book content from: {book_url}")
    
    try:
        response = requests.get(book_url, timeout=15)
        response.raise_for_status()
        
        content = response.text
        
        # Truncate if too long
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated for memory efficiency...]"
        
        logger.info(f"Fetched {len(content)} characters from book")
        return content
        
    except Exception as e:
        logger.error(f"Failed to fetch book content: {e}")
        return f"Error fetching book content: {str(e)}"


@tool
def search_openlibrary_books(query: str, max_results: int = 5) -> str:
    """Search Open Library for books related to a topic.
    
    Args:
        query: Search term for books
        max_results: Maximum number of results to return
    
    Returns:
        JSON string with book information
    """
    logger.debug(f"Searching Open Library for: {query}")
    
    try:
        url = "https://openlibrary.org/search.json"
        params = {
            "q": query,
            "limit": max_results,
            "fields": "key,title,author_name,first_publish_year,subject,ia"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        books = []
        
        for book in data.get("docs", []):
            # Check if book has Internet Archive access
            ia_id = book.get("ia")
            if ia_id and isinstance(ia_id, list) and len(ia_id) > 0:
                ia_url = f"https://archive.org/stream/{ia_id[0]}/{ia_id[0]}_djvu.txt"
            else:
                ia_url = None
            
            book_info = {
                "key": book.get("key"),
                "title": book.get("title", "Unknown Title"),
                "authors": book.get("author_name", ["Unknown Author"]),
                "first_publish_year": book.get("first_publish_year"),
                "subjects": book.get("subject", [])[:5],  # Limit subjects
                "ia_url": ia_url,
                "openlibrary_url": f"https://openlibrary.org{book.get('key', '')}" if book.get('key') else None
            }
            books.append(book_info)
        
        result = {
            "query": query,
            "total_found": data.get("numFound", 0),
            "books": books,
            "success": True
        }
        
        logger.info(f"Found {len(books)} books in Open Library for: {query}")
        return str(result)
        
    except Exception as e:
        logger.error(f"Failed to search Open Library: {e}")
        return str({"error": str(e), "success": False})
