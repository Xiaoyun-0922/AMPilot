"""BLAST search engine wrapper."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil

from ..ingest_normalize.schemas import Sequence
from .schemas import SearchHit, SearchResult, SearchType, DatabaseInfo


class BlastSearchEngine:
    """BLAST+ search engine wrapper."""
    
    def __init__(self, blast_bin_path: Optional[str] = None):
        """
        Initialize BLAST engine.
        
        Args:
            blast_bin_path: Path to BLAST+ binaries directory
        """
        self.blast_bin_path = blast_bin_path
        self._check_blast_installation()
    
    def _check_blast_installation(self):
        """Check if BLAST+ is installed and accessible."""
        try:
            cmd = "blastp" if self.blast_bin_path is None else f"{self.blast_bin_path}/blastp"
            result = subprocess.run([cmd, "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("BLAST+ not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError("BLAST+ not installed or not in PATH")
    
    def search(self,
               query_sequences: List[Sequence],
               database_path: str,
               search_type: SearchType = SearchType.BLASTP,
               evalue_threshold: float = 0.001,
               max_hits: int = 100,
               num_threads: int = 1,
               **kwargs) -> List[SearchResult]:
        """
        Perform BLAST search.
        
        Args:
            query_sequences: List of query sequences
            database_path: Path to BLAST database
            search_type: Type of BLAST search
            evalue_threshold: E-value threshold
            max_hits: Maximum number of hits per query
            num_threads: Number of threads to use
            **kwargs: Additional BLAST parameters
            
        Returns:
            List of SearchResult objects
        """
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_file:
            query_path = query_file.name
            self._write_fasta(query_sequences, query_file)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Build BLAST command
            blast_cmd = self._build_blast_command(
                search_type, query_path, database_path, output_path,
                evalue_threshold, max_hits, num_threads, **kwargs
            )
            
            # Run BLAST
            result = subprocess.run(blast_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"BLAST failed: {result.stderr}")
            
            # Parse results
            search_results = self._parse_blast_xml(output_path, query_sequences)
            
            return search_results
            
        finally:
            # Clean up temporary files
            Path(query_path).unlink(missing_ok=True)
            Path(output_path).unlink(missing_ok=True)
    
    def _build_blast_command(self, search_type: SearchType, query_path: str,
                           database_path: str, output_path: str,
                           evalue_threshold: float, max_hits: int,
                           num_threads: int, **kwargs) -> List[str]:
        """Build BLAST command line."""
        
        blast_program = search_type.value
        if self.blast_bin_path:
            blast_program = f"{self.blast_bin_path}/{blast_program}"
        
        cmd = [
            blast_program,
            "-query", query_path,
            "-db", database_path,
            "-out", output_path,
            "-outfmt", "5",  # XML format
            "-evalue", str(evalue_threshold),
            "-max_target_seqs", str(max_hits),
            "-num_threads", str(num_threads)
        ]
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key.startswith("blast_"):
                param_name = key[6:]  # Remove "blast_" prefix
                cmd.extend([f"-{param_name}", str(value)])
        
        return cmd
    
    def _write_fasta(self, sequences: List[Sequence], file_handle):
        """Write sequences to FASTA file."""
        for seq in sequences:
            file_handle.write(f">{seq.id}")
            if seq.description:
                file_handle.write(f" {seq.description}")
            file_handle.write(f"\n{seq.sequence}\n")
    
    def _parse_blast_xml(self, xml_path: str, query_sequences: List[Sequence]) -> List[SearchResult]:
        """Parse BLAST XML output."""
        results = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Create lookup for query sequences
            query_lookup = {seq.id: seq for seq in query_sequences}
            
            for iteration in root.findall(".//Iteration"):
                query_id = iteration.find("Iteration_query-def").text
                query_len = int(iteration.find("Iteration_query-len").text)
                
                hits = []
                for hit in iteration.findall(".//Hit"):
                    hit_id = hit.find("Hit_id").text
                    hit_def = hit.find("Hit_def").text
                    
                    # Get best HSP (High-scoring Segment Pair)
                    hsp = hit.find(".//Hsp")
                    if hsp is not None:
                        search_hit = self._parse_hsp(hsp, query_id, hit_id, hit_def)
                        hits.append(search_hit)
                
                # Create search result
                search_result = SearchResult(
                    query_id=query_id,
                    query_length=query_len,
                    hits=hits,
                    total_hits=len(hits),
                    search_params={}  # TODO: Add search parameters
                )
                results.append(search_result)
            
            return results
            
        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse BLAST XML output: {e}")
    
    def _parse_hsp(self, hsp_element, query_id: str, hit_id: str, hit_def: str) -> SearchHit:
        """Parse HSP element to SearchHit."""
        
        evalue = float(hsp_element.find("Hsp_evalue").text)
        bit_score = float(hsp_element.find("Hsp_bit-score").text)
        identity = int(hsp_element.find("Hsp_identity").text)
        align_len = int(hsp_element.find("Hsp_align-len").text)
        query_start = int(hsp_element.find("Hsp_query-from").text)
        query_end = int(hsp_element.find("Hsp_query-to").text)
        hit_start = int(hsp_element.find("Hsp_hit-from").text)
        hit_end = int(hsp_element.find("Hsp_hit-to").text)
        
        # Calculate identity percentage
        identity_pct = (identity / align_len) * 100 if align_len > 0 else 0
        
        # Calculate coverage (approximate)
        query_coverage = ((query_end - query_start + 1) / align_len) * 100 if align_len > 0 else 0
        target_coverage = ((hit_end - hit_start + 1) / align_len) * 100 if align_len > 0 else 0
        
        return SearchHit(
            target_id=hit_id,
            target_description=hit_def,
            query_id=query_id,
            evalue=evalue,
            bit_score=bit_score,
            identity=identity_pct,
            similarity=None,  # Not directly available in BLAST
            query_coverage=query_coverage,
            target_coverage=target_coverage,
            alignment_length=align_len,
            query_start=query_start,
            query_end=query_end,
            target_start=hit_start,
            target_end=hit_end,
            query_sequence=hsp_element.find("Hsp_qseq").text if hsp_element.find("Hsp_qseq") is not None else None,
            target_sequence=hsp_element.find("Hsp_hseq").text if hsp_element.find("Hsp_hseq") is not None else None,
            alignment_string=hsp_element.find("Hsp_midline").text if hsp_element.find("Hsp_midline") is not None else None
        )
