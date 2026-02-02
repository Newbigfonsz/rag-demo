"""
Markdown-aware document chunker.

Strategy:
1. Split by markdown headers first (natural semantic boundaries)
2. If sections are too long, use recursive splitting with overlap
3. Preserve metadata (source file, header hierarchy) for citations
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class Chunk:
    """A chunk of text with metadata for retrieval and citation."""
    
    content: str
    source_file: str
    headers: list[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    
    @property
    def metadata(self) -> dict:
        """Return metadata dict for vector DB storage."""
        return {
            "source_file": self.source_file,
            "headers": " > ".join(self.headers) if self.headers else "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }
    
    @property
    def citation(self) -> str:
        """Return a citation string for this chunk."""
        header_path = " > ".join(self.headers) if self.headers else "Introduction"
        return f"{self.source_file}: {header_path}"
    
    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(source={self.source_file}, headers={self.headers}, preview='{preview}')"


class MarkdownChunker:
    """
    Chunks markdown documents by headers with recursive fallback.
    
    Design decisions (see docs/decisions.md ADR-003):
    - Headers provide natural semantic boundaries
    - Recursive splitting handles long sections
    - Overlap prevents losing context at boundaries
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Regex patterns for markdown headers
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def chunk_file(self, file_path: Path) -> list[Chunk]:
        """Chunk a single markdown file."""
        content = file_path.read_text(encoding="utf-8")
        source_name = file_path.stem  # filename without extension
        
        return self.chunk_text(content, source_name)
    
    def chunk_text(self, text: str, source_name: str) -> list[Chunk]:
        """Chunk markdown text into semantic sections."""
        sections = self._split_by_headers(text)
        chunks = []
        
        for section_headers, section_content in sections:
            # Skip empty sections
            if not section_content.strip():
                continue
            
            # If section fits in one chunk, use it directly
            if len(section_content) <= self.chunk_size:
                chunks.append(Chunk(
                    content=section_content.strip(),
                    source_file=source_name,
                    headers=section_headers,
                ))
            else:
                # Section too long, split recursively
                sub_chunks = self._recursive_split(section_content)
                for sub_content in sub_chunks:
                    if sub_content.strip():
                        chunks.append(Chunk(
                            content=sub_content.strip(),
                            source_file=source_name,
                            headers=section_headers,
                        ))
        
        # Add index information
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total
        
        return chunks
    
    def _split_by_headers(self, text: str) -> list[tuple[list[str], str]]:
        """
        Split text by markdown headers, tracking header hierarchy.
        
        Returns list of (headers, content) tuples where headers is the
        hierarchy of headers leading to this section.
        """
        sections = []
        current_headers = []
        current_content = []
        last_header_level = 0
        
        lines = text.split('\n')
        
        for line in lines:
            header_match = self.header_pattern.match(line)
            
            if header_match:
                # Save previous section
                if current_content:
                    content_text = '\n'.join(current_content)
                    sections.append((current_headers.copy(), content_text))
                    current_content = []
                
                # Update header hierarchy
                level = len(header_match.group(1))  # Number of # symbols
                header_text = header_match.group(2).strip()
                
                # Adjust hierarchy based on header level
                if level <= last_header_level:
                    # Going up or same level - trim hierarchy
                    current_headers = current_headers[:level-1]
                
                current_headers.append(header_text)
                last_header_level = level
                
                # Include header in content for context
                current_content.append(line)
            else:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            content_text = '\n'.join(current_content)
            sections.append((current_headers.copy(), content_text))
        
        return sections
    
    def _recursive_split(self, text: str) -> list[str]:
        """
        Recursively split text that's too long.
        
        Split hierarchy:
        1. Double newlines (paragraphs)
        2. Single newlines
        3. Sentences (. ! ?)
        4. Hard split at chunk_size
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        
        # Try splitting by paragraphs first
        separators = ['\n\n', '\n', '. ', '! ', '? ']
        
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                
                for part in parts:
                    # Add separator back (except for paragraph breaks)
                    part_with_sep = part + (separator if separator not in ['\n\n', '\n'] else '\n')
                    
                    if len(current_chunk) + len(part_with_sep) <= self.chunk_size:
                        current_chunk += part_with_sep
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        # Handle overlap
                        if self.chunk_overlap > 0 and current_chunk:
                            overlap_text = current_chunk[-self.chunk_overlap:]
                            current_chunk = overlap_text + part_with_sep
                        else:
                            current_chunk = part_with_sep
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if chunks:
                    return chunks
        
        # Hard split as last resort
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def chunk_directory(self, dir_path: Path) -> Generator[Chunk, None, None]:
        """Chunk all markdown files in a directory (recursive)."""
        for file_path in dir_path.rglob("*.md"):
            chunks = self.chunk_file(file_path)
            yield from chunks


def chunk_documents(
    input_dir: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Convenience function to chunk all documents in a directory.
    
    Args:
        input_dir: Directory containing markdown files
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Chunk objects
    """
    chunker = MarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return list(chunker.chunk_directory(input_dir))


if __name__ == "__main__":
    # Quick test
    from src.config import settings
    
    chunker = MarkdownChunker()
    chunks = list(chunker.chunk_directory(settings.data_raw_dir))
    
    print(f"Created {len(chunks)} chunks from documents")
    print(f"\nSample chunk:")
    print(f"  Source: {chunks[0].source_file}")
    print(f"  Headers: {chunks[0].headers}")
    print(f"  Content preview: {chunks[0].content[:200]}...")
