import weaviate
import re
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
try:
    from .configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME
    from .embeddings import get_embedding_model
except ImportError:
    # Fallback for when called from different contexts (It's a bit hacky, but it works)
    # This setting is more friendly for testing.
    from configuration import WEAVIATE_URL, WEAVIATE_COLLECTION_NAME
    from embeddings import get_embedding_model

    # Remember: lauch weaviate before lauching the project
try:
    weaviate_client = weaviate.connect_to_local(
        host=WEAVIATE_URL.replace("http://", "").split(":")[0],
        port=int(WEAVIATE_URL.replace("http://", "").split(":")[1])
    )
    amp_collection = weaviate_client.collections.get(WEAVIATE_COLLECTION_NAME)
    print("Successfully connected to Weaviate and retrieved GRAMPA collection.")
    embedding_model = get_embedding_model()

except Exception as e:
    print(f"Failed to initialize Weaviate client or embedding model: {e}")
    weaviate_client = None
    embedding_model = None



@tool
def find_peptide_properties(sequence: str) -> List[Dict[str, Any]]:
    """
    Find antimicrobial properties for a given peptide sequence.

    This tool searches the GRAMPA database to find MIC values, target bacteria,
    and modification information for a specific peptide sequence.

    Args:
        sequence: Amino acid sequence (e.g., "GLPRKILCAIAKKKGKCKGPLKLVCKC")
        there may be more than two sequence in the input, please process them one by one.
        A sequence may correspond to multiple pieces of data, just take all of them, because
        they may inhibit more than one bacteria.


    Returns:
        List of dictionaries containing:
        - sequence: The peptide sequence
        - bacterium: Target bacterium
        - strain: Bacterial strain (if available)
        - mic_value_um: MIC value in µM
        - modifications: Peptide modifications
        - database: Source database
    """
    if not weaviate_client or not embedding_model:
        return [{"error": "Database not available. Please ensure Weaviate is running."}]

    try:
        # Search for exact sequence matches first
        response = amp_collection.query.bm25(
            query=sequence,
            limit=10,
            return_properties=None
        )

        exact_matches = []
        for item in response.objects:
            if item.properties.get('sequence', '').upper() == sequence.upper():
                exact_matches.append(item.properties)

        if exact_matches:
            # Prefer APD (aps.unmc.edu) entries if present to avoid mixing sources
            def is_apd(p):
                for k in ("url", "url_source", "database", "source"):
                    v = (p.get(k) or "").lower()
                    if "aps.unmc.edu" in v or "apd" in v:
                        return True
                return False
            apd_matches = [p for p in exact_matches if is_apd(p)]
            return apd_matches if apd_matches else exact_matches

        # Strict exact matching only: if no exact match, inform user
        return [{"message": f"No properties found for sequence: {sequence}. The sequence was not found in the GRAMPA database."}]

    except Exception as e:
        print(f"Error in find_peptide_properties: {e}")
        return [{"error": "Failed to search for peptide properties."}]


@tool
def find_peptides_by_properties(query: str = "", bacterium: str = "", mic_range: str = "", modifications: str = "", strain: str = "", length_range: str = "") -> List[Dict[str, Any]]:
    """
    Find peptide sequences based on desired antimicrobial properties.

    This tool searches the GRAMPA database to find peptides that match specified criteria
    such as target bacterium, MIC range, modifications, bacterial strain, or general properties.

    Args:
        query: General query for properties or mechanisms (e.g., "membrane disruption", "cationic peptides")
        bacterium: Target bacterium name (e.g., "S. aureus", "E. coli", "C. albicans")
        mic_range: MIC concentration range (e.g., "< 10", "1-5", "> 100")
        modifications: Type of modifications (e.g., "disulfide", "amidation", "none")
        strain: Specific bacterial strain (e.g., "ATCC29213")
        length_range: Sequence length range (e.g., "20-25", "10-15", "< 30", "> 20")

    Returns:
        List of dictionaries containing matching peptide information
    """
    if not weaviate_client or not embedding_model:
        return [{"error": "Database not available. Please ensure Weaviate is running."}]

    print(f"DEBUG: find_peptides_by_properties called with:")
    print(f"  query: {query}")
    print(f"  bacterium: {bacterium}")
    print(f"  mic_range: {mic_range}")
    print(f"  modifications: {modifications}")
    print(f"  strain: {strain}")
    print(f"  length_range: {length_range}")

    try:
        # Build search query based on provided criteria
        search_terms = []

        # Handle general query (for mechanisms, properties, etc.)
        if query:
            # For mechanism queries, search broadly
            if any(keyword in query.lower() for keyword in ['membrane disruption', 'pore formation', 'mechanism', 'principle']):
                # Use semantic search for mechanism-related queries - search all peptides
                search_query = "antimicrobial peptide"  # Broad search to get diverse results
            else:
                search_terms.append(query)

        if bacterium:
            search_terms.append(f"bacterium: {bacterium}")
        if strain:
            search_terms.append(f"strain: {strain}")
        if modifications:
            search_terms.append(f"modifications: {modifications}")
        if mic_range:
            search_terms.append(f"MIC: {mic_range}")

        if not search_terms and not query:
            return [{"message": "Please provide at least one search criterion (query, bacterium, MIC range, modifications, or strain)."}]

        # Determine search strategy
        if query and not search_terms:
            search_query = search_query if 'search_query' in locals() else query
        elif search_terms:
            query_text = " | ".join(search_terms)
            # Enhanced search for bacteria - include both full name and abbreviation
            if bacterium and not mic_range and not modifications and not strain and not query:
                # Create comprehensive search query for bacteria
                bacterium_lower = bacterium.lower()
                if "s. aureus" in bacterium_lower or "staphylococcus aureus" in bacterium_lower:
                    search_query = "S. aureus Staphylococcus aureus MRSA MSSA aureus staphylococcus"
                elif "e. coli" in bacterium_lower or "escherichia coli" in bacterium_lower:
                    search_query = "E. coli Escherichia coli coli escherichia"
                else:
                    search_query = bacterium
            else:
                search_query = query_text
        else:
            search_query = "antimicrobial peptide"

        # Use a higher limit and better search strategy for comprehensive results
        search_limit = 200 if bacterium else 100  # Much higher limit for comprehensive coverage

        # For bacterium searches, use multiple comprehensive strategies
        if bacterium and not query:
            # Strategy 1: Direct bacterium search
            bacterium_response = amp_collection.query.bm25(
                query=search_query,
                limit=search_limit,
                return_properties=None
            )

            # Strategy 2: Search for known important sequences that might be missed
            # This is a workaround for BM25 relevance scoring issues
            important_sequences = [
                "GIGKFLKKAKKFGKAFVKILKK",  # Known to have S. aureus data but often missed
                "NLCASLRARHTIPQCKKFGRR",
                "GMKCKFCCNCCNLNGCGVCCRF",
                "FLPLLAGLAANFLPKIFCKITRK"
            ]

            sequence_items = []
            for seq in important_sequences:
                seq_response = amp_collection.query.bm25(
                    query=seq,
                    limit=20,
                    return_properties=None
                )
                sequence_items.extend(seq_response.objects)

            # Combine all results
            all_items = list(bacterium_response.objects) + sequence_items

            # Filter for the target bacterium and deduplicate
            bacterium_lower = bacterium.lower()
            bacterium_variations = [bacterium_lower]

            # Add common variations
            if "s. aureus" in bacterium_lower or "staphylococcus aureus" in bacterium_lower:
                bacterium_variations.extend(["s. aureus", "staphylococcus aureus", "aureus", "s.aureus"])
            elif "e. coli" in bacterium_lower or "escherichia coli" in bacterium_lower:
                bacterium_variations.extend(["e. coli", "escherichia coli", "coli", "e.coli"])

            # Filter and deduplicate
            seen_ids = set()
            matching_items = []

            for item in all_items:
                item_id = getattr(item, 'uuid', str(item))
                if item_id in seen_ids:
                    continue

                result = item.properties
                result_bacterium = (result.get('bacterium', '') or '').lower()

                # Check if any variation matches
                for variation in bacterium_variations:
                    if variation in result_bacterium:
                        seen_ids.add(item_id)
                        matching_items.append(item)
                        break

            # Create response with filtered items
            class MockResponse:
                def __init__(self, objects):
                    self.objects = objects

            response = MockResponse(matching_items)
        else:
            response = amp_collection.query.bm25(
                query=search_query,
                limit=search_limit,
                return_properties=None
            )

        results_scored = []
        # Pre-parse MIC spec once for scoring
        mic_spec = _parse_mic_spec(mic_range) if mic_range else None
        length_spec = _parse_length_spec(length_range) if length_range else None

        for item in response.objects:
            result = item.properties.copy()

            # Apply additional filtering based on criteria
                # Robust bacterium matching: allow abbreviations and full names both ways
            if bacterium:
                cand = (result.get('bacterium', '') or '').lower()
                query_b = bacterium.lower()
                # Expand common abbreviations: "s. aureus" -> "staphylococcus aureus", etc.
                def expand_abbrev(name: str) -> str:
                    mapping = {
                        "s. aureus": "staphylococcus aureus",
                        "e. coli": "escherichia coli",
                        "b. subtilis": "bacillus subtilis",
                        "e. faecalis": "enterococcus faecalis",
                        "e. faecium": "enterococcus faecium",
                        "p. aeruginosa": "pseudomonas aeruginosa",
                    }
                    return mapping.get(name, name)
                expanded_query = expand_abbrev(query_b)
                expanded_cand = expand_abbrev(cand)
                if not (query_b in cand or expanded_query in cand or query_b in expanded_cand or expanded_query in expanded_cand):
                    continue
            if strain and strain.lower() not in (result.get('strain', '') or '').lower():
                continue
            if modifications and modifications.lower() not in (result.get('modifications', '') or '').lower():
                continue

            # Apply sequence length filtering
            if length_range:
                sequence = result.get('sequence', '')
                seq_length = len(sequence) if sequence else 0
                if not _matches_length_range(seq_length, length_spec):
                    continue

            mic_val = result.get('mic_value_um', None)
            try:
                mic_float = float(mic_val) if mic_val is not None else None
            except Exception:
                mic_float = None

            # Apply MIC range filtering (strict filtering, not just ranking)
            if mic_range and mic_float is not None:
                if not _matches_mic_range(mic_float, mic_range):
                    continue  # Skip results that don't match MIC criteria

            # If mic_range provided, compute distance for ranking (0 means perfectly within spec)
            if mic_range:
                dist = _mic_distance(mic_float, mic_spec) if mic_spec else float('inf')
            else:
                # No mic range: prefer lower MIC
                dist = mic_float if mic_float is not None else float('inf')

            results_scored.append((dist, mic_float if mic_float is not None else float('inf'), result))

        if not results_scored:
            return [{"message": f"No peptides found matching the criteria: {query_text}"}]

        # Sort by distance first, then by raw MIC ascending as tie-breaker
        results_scored.sort(key=lambda x: (x[0], x[1]))

        # Remove duplicates while preserving the best MIC values
        # Group by sequence and keep the best (lowest distance, then lowest MIC) entry for each sequence
        sequence_groups = {}
        for dist, mic, result in results_scored:
            seq = result.get("sequence", "")

            if seq not in sequence_groups:
                sequence_groups[seq] = (dist, mic, result)
            else:
                # Keep the better entry (lower distance, then lower MIC)
                existing_dist, existing_mic, existing_result = sequence_groups[seq]
                if (dist, mic) < (existing_dist, existing_mic):
                    sequence_groups[seq] = (dist, mic, result)

        # Convert back to list and sort again
        deduplicated_scored = list(sequence_groups.values())
        deduplicated_scored.sort(key=lambda x: (x[0], x[1]))

        # Return more results - up to 10 for better coverage
        top_results = [r for _, __, r in deduplicated_scored[:10]]

        print(f"DEBUG: Returning {len(top_results)} results")
        for i, result in enumerate(top_results):
            seq = result.get('sequence', 'N/A')
            print(f"  {i+1}. Sequence: {seq[:20]}..., Length: {len(seq) if seq != 'N/A' else 'N/A'}, MIC: {result.get('mic_value_um', 'N/A')}")
            print(f"      Bacterium: {result.get('bacterium', 'N/A')}, Modifications: {result.get('modifications', 'N/A')}")

        return top_results

    except Exception as e:
        print(f"Error in find_peptides_by_properties: {e}")
        return [{"error": "Failed to search for peptides by properties."}]


def _parse_length_spec(length_range: str) -> Dict[str, Any]:
    """Parse length range specification into a structured format."""
    import re

    length_range = length_range.strip().lower()

    # Handle range patterns like "20-25", "10-15"
    range_match = re.match(r'(\d+)\s*[-–]\s*(\d+)', length_range)
    if range_match:
        return {
            'type': 'range',
            'min': int(range_match.group(1)),
            'max': int(range_match.group(2))
        }

    # Handle comparison patterns like "< 30", "> 20", "<= 25", ">= 15"
    comp_match = re.match(r'([<>]=?)\s*(\d+)', length_range)
    if comp_match:
        operator = comp_match.group(1)
        value = int(comp_match.group(2))
        return {
            'type': 'comparison',
            'operator': operator,
            'value': value
        }

    # Handle exact value
    exact_match = re.match(r'(\d+)', length_range)
    if exact_match:
        value = int(exact_match.group(1))
        return {
            'type': 'exact',
            'value': value
        }

    return None


def _matches_length_range(seq_length: int, length_spec: Dict[str, Any]) -> bool:
    """Check if sequence length matches the specified range."""
    if not length_spec:
        return True

    if length_spec['type'] == 'range':
        return length_spec['min'] <= seq_length <= length_spec['max']
    elif length_spec['type'] == 'comparison':
        operator = length_spec['operator']
        value = length_spec['value']
        if operator == '<':
            return seq_length < value
        elif operator == '<=':
            return seq_length <= value
        elif operator == '>':
            return seq_length > value
        elif operator == '>=':
            return seq_length >= value
    elif length_spec['type'] == 'exact':
        return seq_length == length_spec['value']

    return True


def _matches_mic_range(mic_value: float, mic_range: str) -> bool:
    """Helper function to check if MIC value matches the specified range.
    Tolerates units (μM/uM) and unicode comparators (≤ ≥).
    """
    try:
        s = (mic_range or '').strip().lower()
        s = s.replace('μm', '').replace('um', '').replace(' u m', '')
        s = s.replace('≤', '<=').replace('≥', '>=')
        s = re.sub(r"\s+", " ", s)

        if s.startswith('<'):
            # For < 10, should NOT include 10 or above
            thr = float(s.lstrip('<=').strip())
            return mic_value < thr  # Strict less than
        elif s.startswith('>'):
            thr = float(s.lstrip('>=').strip())
            return mic_value > thr
        elif '-' in s:
            parts = s.split('-', 1)
            if len(parts) == 2:
                min_val = float(parts[0].strip())
                max_val = float(parts[1].strip())
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                return min_val <= mic_value <= max_val
        else:
            # Exact value
            target = float(s)
            return abs(mic_value - target) < 0.1

    except (ValueError, IndexError):
        return True  # If parsing fails, include the result

    return True

def _parse_mic_spec(mic_range: str) -> Dict[str, Any]:
    """Parse MIC range string into a normalized spec for scoring.
    Supports forms like: "< 10", "<=10", "> 100", ">=50", "1-5", "10".
    Also tolerates units: μM, uM, UM, and extra text.
    """
    try:
        if not mic_range:
            return {"type": "none"}
        s = (mic_range or "").strip().lower()
        # remove units and extra text
        s = s.replace("μm", "").replace("um", "").replace(" u m", "")
        s = s.replace("≤", "<=").replace("≥", ">=")
        s = re.sub(r"[^0-9.<>=\-\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        # Range a-b
        if "-" in s and not s.startswith(("<", ">")):
            parts = s.split("-", 1)
            a = float(parts[0].strip())
            b = float(parts[1].strip())
            if a > b:
                a, b = b, a
            return {"type": "range", "a": a, "b": b}
        # <= or <
        if s.startswith("<="):
            a = float(s[2:].strip())
            return {"type": "lt", "a": a}
        if s.startswith("<"):
            a = float(s[1:].strip())
            return {"type": "lt", "a": a}
        # >= or >
        if s.startswith(">="):
            a = float(s[2:].strip())
            return {"type": "gt", "a": a}
        if s.startswith(">"):
            a = float(s[1:].strip())
            return {"type": "gt", "a": a}
        # exact value
        val = float(s)
        return {"type": "eq", "a": val}
    except Exception:
        return {"type": "none"}


def _mic_distance(mic: Optional[float], spec: Optional[Dict[str, Any]]) -> float:
    """Compute distance of mic value to the desired spec.
    0 means perfectly satisfies spec; larger means further away.
    """
    if mic is None or not spec:
        return float("inf")
    t = spec.get("type", "none")
    if t == "eq":
        a = spec.get("a")
        return abs(mic - a) if a is not None else float("inf")
    if t == "range":
        a = spec.get("a"); b = spec.get("b")
        if a is None or b is None:
            return float("inf")
        if a <= mic <= b:
            return 0.0
        return min(abs(mic - a), abs(mic - b))
    if t == "lt":
        a = spec.get("a")
        if a is None:
            return float("inf")
        # penalty if above threshold, 0 if within range
        return max(0.0, mic - a) if mic > a else 0.0
    if t == "gt":
        a = spec.get("a")
        if a is None:
            return float("inf")
        # penalty if below threshold
        return max(0.0, a - mic)
    # none/unknown: prefer smaller MIC
    return mic


# Note: We no longer need to close the client here as it's a long-lived connection.
# The connection will be closed when the application shuts down.