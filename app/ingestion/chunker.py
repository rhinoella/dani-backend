from __future__ import annotations

import re
from typing import Dict, List

import tiktoken
import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


class TokenChunker:
    """
    Token-based text chunker optimized for meeting transcripts.
    
    Defaults optimized for conversational content:
    - chunk_size=350: Smaller chunks capture atomic ideas better in conversations
    - overlap=100: ~28% overlap prevents context breaks mid-thought
    - Speaker-aware chunking respects dialogue boundaries
    """
    def __init__(
        self, 
        chunk_size: int = 350,   # Reduced from 512 for better semantic coherence
        overlap: int = 100,      # Increased from 64 (~28% overlap)
        max_tokens: int = 8192,
        sentence_aware: bool = True,
        speaker_aware: bool = True
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_tokens = max_tokens
        self.sentence_aware = sentence_aware
        self.speaker_aware = speaker_aware
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, record: Dict) -> List[Dict]:
        """
        Chunk a single normalized record into token windows.
        Respects sentence boundaries when sentence_aware=True.
        """
        text = record["text"]
        metadata = record.get("metadata", {})
        
        if self.sentence_aware:
            return self._chunk_sentence_aware(text, metadata)
        else:
            return self._chunk_token_based(text, metadata)
    
    def chunk_with_speakers(self, sentences: List[Dict], metadata: Dict) -> List[Dict]:
        """
        Chunk sentences while respecting speaker boundaries.
        Each sentence should have 'text' and optionally 'speaker_name' or 'speaker'.
        Chunks will not break in the middle of one speaker's continuous dialogue.
        """
        if not sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_speakers = set()
        current_tokens = 0
        chunk_index = 0
        
        for i, sentence in enumerate(sentences):
            text = sentence.get("text", "").strip()
            if not text:
                continue
            
            speaker = sentence.get("speaker_name") or sentence.get("speaker") or "Unknown"
            sentence_tokens = len(self.encoder.encode(text))
            
            # Check if adding this sentence would exceed chunk_size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                # If speaker_aware is enabled and this is the same speaker, allow exceeding slightly
                # unless it's WAY over (2x chunk_size)
                is_same_speaker = speaker in current_speakers
                would_exceed_hard_limit = current_tokens + sentence_tokens > self.chunk_size * 2
                
                if not self.speaker_aware or not is_same_speaker or would_exceed_hard_limit:
                    # Save current chunk
                    chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
                    chunks.append(self._create_chunk(
                        chunk_text, 
                        {**metadata, "speakers": list(current_speakers)},
                        chunk_index
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap (last few sentences)
                    overlap_sentences = self._get_overlap_sentences_with_speakers(current_chunk_sentences)
                    current_chunk_sentences = overlap_sentences + [sentence]
                    current_speakers = {s.get("speaker_name") or s.get("speaker") or "Unknown" for s in current_chunk_sentences}
                    current_tokens = sum(len(self.encoder.encode(s["text"])) for s in current_chunk_sentences)
                    continue
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_speakers.add(speaker)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(s["text"] for s in current_chunk_sentences)
            chunks.append(self._create_chunk(
                chunk_text,
                {**metadata, "speakers": list(current_speakers)},
                chunk_index
            ))
        
        return chunks
    
    def _get_overlap_sentences_with_speakers(self, sentences: List[Dict]) -> List[Dict]:
        """Get last sentences that fit within overlap token limit."""
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            text = sentence.get("text", "")
            sentence_tokens = len(self.encoder.encode(text))
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _chunk_sentence_aware(self, text: str, metadata: Dict) -> List[Dict]:
        """Chunk text while respecting sentence boundaries."""
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self.encoder.encode(sentence)
            sentence_token_count = len(sentence_tokens)
            
            # If adding this sentence would exceed chunk_size
            if current_tokens + sentence_token_count > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_sentences(current_chunk)
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_tokens = len(self.encoder.encode(" ".join(current_chunk)))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_token_count
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> str:
        """Get last sentences that fit within overlap token limit."""
        overlap_sentences = []
        overlap_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = len(self.encoder.encode(sentence))
            if overlap_tokens + sentence_tokens <= self.overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return " ".join(overlap_sentences)
    
    def _chunk_token_based(self, text: str, metadata: Dict) -> List[Dict]:
        """Original token-based chunking (fallback)."""
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Safety check: ensure chunk doesn't exceed model's max context
            if len(chunk_tokens) > self.max_tokens:
                chunk_tokens = chunk_tokens[:self.max_tokens]
            
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))

            start += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks
    
    def _create_chunk(self, text: str, metadata: Dict, chunk_index: int) -> Dict:
        """Create a chunk dictionary with metadata."""
        tokens = self.encoder.encode(text)
        return {
            "text": text,
            "metadata": {
                **metadata,
                "chunk_index": chunk_index,
                "token_count": len(tokens),
            },
        }
