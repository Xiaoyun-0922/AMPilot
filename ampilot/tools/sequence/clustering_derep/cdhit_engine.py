"""CD-HIT clustering engine wrapper."""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

from ..ingest_normalize.schemas import Sequence
from ..homology_search.schemas import HomologSet, SearchResult
from .schemas import Cluster, ClusterMember, ClusteringMethod, ClusteringMode


class CDHitEngine:
    """CD-HIT clustering engine wrapper."""
    
    def __init__(self, cdhit_bin_path: Optional[str] = None):
        """
        Initialize CD-HIT engine.
        
        Args:
            cdhit_bin_path: Path to CD-HIT binaries directory
        """
        self.cdhit_bin_path = cdhit_bin_path
        self._check_cdhit_installation()
    
    def _check_cdhit_installation(self):
        """Check if CD-HIT is installed and accessible."""
        try:
            cmd = "cd-hit" if self.cdhit_bin_path is None else f"{self.cdhit_bin_path}/cd-hit"
            result = subprocess.run([cmd, "-h"], 
                                  capture_output=True, text=True, timeout=10)
            # CD-HIT returns non-zero exit code even for help
            if "CD-HIT" not in result.stderr:
                raise RuntimeError("CD-HIT not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError("CD-HIT not installed or not in PATH")
    
    def cluster_sequences(self,
                         sequences: List[Sequence],
                         identity_threshold: float = 0.9,
                         coverage_threshold: float = 0.8,
                         word_size: Optional[int] = None,
                         num_threads: int = 1,
                         memory_limit: int = 800,  # MB
                         **kwargs) -> List[Cluster]:
        """
        Cluster sequences using CD-HIT.
        
        Args:
            sequences: List of sequences to cluster
            identity_threshold: Identity threshold (0.0-1.0)
            coverage_threshold: Coverage threshold (0.0-1.0)
            word_size: Word size for clustering (auto if None)
            num_threads: Number of threads
            memory_limit: Memory limit in MB
            **kwargs: Additional CD-HIT parameters
            
        Returns:
            List of Cluster objects
        """
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as input_file:
            input_path = input_file.name
            self._write_fasta(sequences, input_file)
        
        output_path = input_path + ".clstr"
        result_path = input_path + ".out"
        
        try:
            # Determine appropriate CD-HIT program and word size
            cdhit_program, auto_word_size = self._select_cdhit_program(identity_threshold)
            word_size = word_size or auto_word_size
            
            # Build CD-HIT command
            cmd = self._build_cdhit_command(
                cdhit_program, input_path, result_path,
                identity_threshold, coverage_threshold,
                word_size, num_threads, memory_limit, **kwargs
            )
            
            # Run CD-HIT
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                raise RuntimeError(f"CD-HIT failed: {result.stderr}")
            
            # Parse clustering results
            clusters = self._parse_cdhit_output(output_path, sequences, identity_threshold, coverage_threshold)
            
            return clusters
            
        finally:
            # Clean up temporary files
            for temp_file in [input_path, output_path, result_path]:
                Path(temp_file).unlink(missing_ok=True)
    
    def _select_cdhit_program(self, identity_threshold: float) -> tuple[str, int]:
        """Select appropriate CD-HIT program and word size based on identity threshold."""
        
        if identity_threshold >= 0.7:
            program = "cd-hit"
            word_size = 5
        elif identity_threshold >= 0.6:
            program = "cd-hit"
            word_size = 4
        elif identity_threshold >= 0.5:
            program = "cd-hit"
            word_size = 3
        elif identity_threshold >= 0.4:
            program = "cd-hit"
            word_size = 2
        else:
            program = "cd-hit-est"  # For very low identity
            word_size = 2
        
        if self.cdhit_bin_path:
            program = f"{self.cdhit_bin_path}/{program}"
        
        return program, word_size
    
    def _build_cdhit_command(self, program: str, input_path: str, output_path: str,
                           identity_threshold: float, coverage_threshold: float,
                           word_size: int, num_threads: int, memory_limit: int,
                           **kwargs) -> List[str]:
        """Build CD-HIT command line."""
        
        cmd = [
            program,
            "-i", input_path,
            "-o", output_path,
            "-c", str(identity_threshold),
            "-aS", str(coverage_threshold),  # Coverage for shorter sequence
            "-n", str(word_size),
            "-T", str(num_threads),
            "-M", str(memory_limit),
            "-d", "0"  # Length of description in .clstr file
        ]
        
        # Add additional parameters
        for key, value in kwargs.items():
            if key.startswith("cdhit_"):
                param_name = key[6:]  # Remove "cdhit_" prefix
                if len(param_name) == 1:
                    cmd.extend([f"-{param_name}", str(value)])
                else:
                    cmd.extend([f"-{param_name}", str(value)])
        
        return cmd
    
    def _write_fasta(self, sequences: List[Sequence], file_handle):
        """Write sequences to FASTA file."""
        for seq in sequences:
            file_handle.write(f">{seq.id}")
            if seq.description:
                file_handle.write(f" {seq.description}")
            file_handle.write(f"\n{seq.sequence}\n")
    
    def _parse_cdhit_output(self, clstr_path: str, original_sequences: List[Sequence],
                          identity_threshold: float, coverage_threshold: float) -> List[Cluster]:
        """Parse CD-HIT .clstr output file."""
        
        clusters = []
        current_cluster = None
        
        # Create sequence lookup
        seq_lookup = {seq.id: seq for seq in original_sequences}
        
        with open(clstr_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('>Cluster'):
                    # Start new cluster
                    if current_cluster is not None:
                        clusters.append(current_cluster)
                    
                    cluster_id = line.split()[1]
                    current_cluster = {
                        'id': cluster_id,
                        'members': [],
                        'representative': None
                    }
                
                elif line and current_cluster is not None:
                    # Parse cluster member
                    member_info = self._parse_cluster_member_line(line, seq_lookup)
                    if member_info:
                        if member_info['is_representative']:
                            current_cluster['representative'] = member_info
                        else:
                            current_cluster['members'].append(member_info)
        
        # Add last cluster
        if current_cluster is not None:
            clusters.append(current_cluster)
        
        # Convert to Cluster objects
        cluster_objects = []
        for cluster_data in clusters:
            if cluster_data['representative']:
                # Create representative ClusterMember
                rep_data = cluster_data['representative']
                representative = ClusterMember(
                    sequence_id=rep_data['sequence_id'],
                    sequence=rep_data['sequence'],
                    length=rep_data['length'],
                    is_representative=True
                )
                
                # Create member ClusterMembers
                members = []
                for member_data in cluster_data['members']:
                    member = ClusterMember(
                        sequence_id=member_data['sequence_id'],
                        sequence=member_data['sequence'],
                        length=member_data['length'],
                        is_representative=False,
                        similarity_to_rep=member_data.get('similarity')
                    )
                    members.append(member)
                
                # Calculate average length
                all_lengths = [representative.length] + [m.length for m in members]
                avg_length = sum(all_lengths) / len(all_lengths)
                
                cluster = Cluster(
                    cluster_id=cluster_data['id'],
                    representative=representative,
                    members=members,
                    size=len(members) + 1,
                    avg_length=avg_length,
                    identity_threshold=identity_threshold,
                    coverage_threshold=coverage_threshold
                )
                cluster_objects.append(cluster)
        
        return cluster_objects
    
    def _parse_cluster_member_line(self, line: str, seq_lookup: Dict[str, Sequence]) -> Optional[Dict[str, Any]]:
        """Parse a single cluster member line from .clstr file."""
        
        # Example line: "0	426aa, >WP_003131952.1... *"
        # Example line: "1	413aa, >WP_003131953.1... at +/99.53%"
        
        try:
            parts = line.split('\t')
            if len(parts) < 2:
                return None
            
            info_part = parts[1]
            
            # Extract sequence ID
            id_match = re.search(r'>([^\s\.]+)', info_part)
            if not id_match:
                return None
            
            seq_id = id_match.group(1)
            
            # Get sequence from lookup
            if seq_id not in seq_lookup:
                return None
            
            sequence = seq_lookup[seq_id]
            
            # Check if representative (ends with *)
            is_representative = info_part.endswith('*')
            
            # Extract similarity if not representative
            similarity = None
            if not is_representative:
                sim_match = re.search(r'at [+-]/(\d+\.?\d*)%', info_part)
                if sim_match:
                    similarity = float(sim_match.group(1))
            
            return {
                'sequence_id': seq_id,
                'sequence': sequence.sequence,
                'length': sequence.length or len(sequence.sequence),
                'is_representative': is_representative,
                'similarity': similarity
            }
            
        except Exception:
            return None
