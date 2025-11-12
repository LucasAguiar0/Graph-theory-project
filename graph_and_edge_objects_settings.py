from typing import Iterable, List, Tuple, Dict, Any
import numpy as np
import copy


class Edge:
    def __init__(self, vertex1: Any, vertex2: Any, weight: float = 1, frequency: int = 1):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight
        self.frequency = frequency

    def __repr__(self):
        parts = []
        if self.weight != 1:
            parts.append(f"weight={self.weight}")
        if self.frequency != 1:
            parts.append(f"frequency={self.frequency}")
        if parts:
            return f"({self.vertex1},{self.vertex2}," + ",".join(parts) + ")"
        return f"({self.vertex1},{self.vertex2})"

    # equality/hash include all fields so Edge objects are 'fully equal' only when identical
    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return (
            self.vertex1 == other.vertex1 and
            self.vertex2 == other.vertex2 and
            self.weight == other.weight and
            self.frequency == other.frequency
        )

    def __hash__(self):
        return hash((self.vertex1, self.vertex2, self.weight, self.frequency))


class Graph:
    def __init__(self,
                 vertices: Iterable = None,
                 edges: Iterable[Edge] = None,
                 directed: bool = False,
                 multigraph: bool = False,
                 pseudograph: bool = False,
                 weighted: bool = False,
                 name = "None"):
        # safe defaults
        self.name = name
        self.vertices = set(vertices) if vertices is not None else set()
        self._raw_edges: List[Edge] = list(edges) if edges is not None else []

        self.sorted_vertices = sorted(list(self.vertices))

        # flags
        self.directed = bool(directed)
        self.multigraph = bool(multigraph)
        self.pseudograph = bool(pseudograph)
        self.weighted = bool(weighted)

        # pseudographs imply multigraphs
        if self.pseudograph:
            self.multigraph = True

        # canonical final edge list (after merging/dedup)
        self.edges: List[Edge] = []

        # build processed edge list from _raw_edges
        self._process_edges()
        
        self.order = len(self.vertices)
        self.size = len(self.edges)
        
        self.adjacency_dict = self.get_adjacency_dict()
        self.adjacency_matrix = None
        self.frequency_dictionaries = {}
        self.get_adjacency_matrix()
        

        self.degree_matrix = None
        self.laplacian_matrix = None
        
        self.n_step_walk_matrix = None
        self.cumulative_n_step_walk_matrix = None

        self.simple_undirected_copy = None
        
    # ---------- internal helpers ----------
    def _canonical_pair(self, a, b):
        """Return (v1, v2) in deterministic canonical order (string order),
           used for undirected edges stored with canonical orientation."""
        # Use string comparison so mixed types behave predictably
        return (a, b) if str(a) <= str(b) else (b, a)

    def _validate_and_prepare_raw(self): 
        """Validate edges (vertices must exist) and apply weighting rule if needed."""
        filtered_edges = []
        for e in self._raw_edges:
            if e.vertex1 not in self.vertices or e.vertex2 not in self.vertices:
                raise ValueError("All edges must contain given vertices (vertex missing).")
            
            # if graph isn't weighted, force weight to 1
            if not self.weighted:
                e.weight = 1

            # if not pseudograph, skip loops instead of raising
            if not self.pseudograph and e.vertex1 == e.vertex2:
                continue  # just ignore this edge instead of raising
            
            filtered_edges.append(e)

        # store only the valid edges
        self._raw_edges = filtered_edges
        
    def _process_edges(self):
        """Process the raw edges and populate self.edges according to flags."""
        # validate & prepare
        self._validate_and_prepare_raw()

        raw = self._raw_edges  # alias

        if self.multigraph:
            # Multigraph: merge duplicates by (endpoints ignoring orientation for undirected) + weight (if weighted)
            edge_map = {}
            for e in raw:
                if self.directed:
                    # directed: (v1, v2, weight?) distinguishes edges
                    key = (e.vertex1, e.vertex2, e.weight) if self.weighted else (e.vertex1, e.vertex2)
                    if key in edge_map:
                        edge_map[key].frequency += e.frequency
                    else:
                        # store a copy
                        edge_map[key] = Edge(e.vertex1, e.vertex2, weight=e.weight, frequency=e.frequency)
                else:
                    # undirected multigraph: group by unordered endpoints + weight (if weighted)
                    endpoint_set = frozenset([e.vertex1, e.vertex2])
                    if self.weighted:
                        key = (endpoint_set, e.weight)
                    else:
                        key = endpoint_set
                    if key in edge_map:
                        edge_map[key].frequency += e.frequency
                    else:
                        # store with canonical orientation for consistent representation
                        v1, v2 = self._canonical_pair(e.vertex1, e.vertex2)
                        edge_map[key] = Edge(v1, v2, weight=e.weight, frequency=e.frequency)

            self.edges = list(edge_map.values())

        else:
            # Simple graph (not a multigraph): keep at most one edge between a pair
            # If multiple candidates exist, keep the one with smallest weight
            unique = {}
            for e in raw:
                if self.directed:
                    key = (e.vertex1, e.vertex2)
                else:
                    key = frozenset([e.vertex1, e.vertex2])

                if key not in unique:
                    # store a copy and ensure frequency = 1 for simple graph
                    if not self.directed:
                        v1, v2 = self._canonical_pair(e.vertex1, e.vertex2)
                    else:
                        v1, v2 = e.vertex1, e.vertex2
                    unique[key] = Edge(v1, v2, weight=e.weight, frequency=1)
                else:
                    # choose the smallest weight
                    if e.weight < unique[key].weight:
                        if not self.directed:
                            v1, v2 = self._canonical_pair(e.vertex1, e.vertex2)
                        else:
                            v1, v2 = e.vertex1, e.vertex2
                        unique[key] = Edge(v1, v2, weight=e.weight, frequency=1)

            self.edges = list(unique.values())

   
    def reset(self):
        self.get_adjacency_dict()
        self.sorted_vertices = sorted(list(self.vertices))
        self.get_adjacency_matrix()
        self.n_step_walk_matrix = None
        self.cumulative_n_step_walk_matrix = None
        self.laplacian_matrix = None
        self.degree_matrix = None
        self.simple_undirected_copy = None

    def add_vertex(self, v):
        """Add a vertex (no reprocessing required for edges)."""
        self.vertices.add(v)
        self.order = len(self.vertices)
        #If a vertex or edge is added, the vertices dictionary and adjacency matrix need to be updated. This is why this variable is set to True.
        self.reset()
        

    def add_edge(self, edge: Edge, validate_vertices: bool = True):
        """Add an edge to the raw list and reprocess everything (simple & safe)."""
        if validate_vertices:
            # optionally add vertices automatically (you can change to raise instead)
            if edge.vertex1 not in self.vertices:
                self.vertices.add(edge.vertex1)
            if edge.vertex2 not in self.vertices:
                self.vertices.add(edge.vertex2)
        self._raw_edges.append(edge)
        self._process_edges()
        self.size = len(self.edges)
        self.reset()
        self.get_all()

    def get_adjacency_dict(self):
        self.adjacency_dict = {}
        for vertex in self.vertices:
            self.adjacency_dict[vertex]=set()
            for edge in self.edges:
                if self.directed == True:
                    if edge.vertex1 == vertex:
                        vertex_weight_frequency = (edge.vertex2, edge.weight, edge.frequency)
                        self.adjacency_dict[vertex].add(vertex_weight_frequency)
                else:
                    if edge.vertex1 == vertex:
                        vertex_weight_frequency = (edge.vertex2, edge.weight, edge.frequency)
                        self.adjacency_dict[vertex].add(vertex_weight_frequency)
                    elif edge.vertex2 == vertex:
                        vertex_weight_frequency = (edge.vertex1, edge.weight, edge.frequency)
                        self.adjacency_dict[vertex].add(vertex_weight_frequency)
        return self.adjacency_dict

    def print_adjacency_dict(self):
        sorted_pairs = sorted(self.adjacency_dict.items())
        for key, vertex_list in sorted_pairs:
            if not (vertex_list == set()):
                print(key+ ": {",end ="")
                for vertex_weight_frequency in sorted(vertex_list,key=lambda x:x[0]):
                    print("\n-"+str(vertex_weight_frequency[0])+":", "weight:", str(vertex_weight_frequency[1])+",", "frequency:", vertex_weight_frequency[2],end="")
                print("\n}\n")

    def degree(self, vertex):
        if self.directed:
            # For directed graphs, degree = in-degree + out-degree
            return self.in_degree(vertex) + self.out_degree(vertex)

        # Undirected (including pseudographs)
        degree_sum = 0

        for neighbor, weight, frequency in self.adjacency_dict[vertex]:
            # Each edge contributes its frequency
            if neighbor == vertex:
                # Loop contributes twice to the degree
                degree_sum += 2 * frequency
            else:
                degree_sum += frequency

        return degree_sum

    def out_degree(self, vertex):
        if not self.directed:
            raise ValueError("This method is only valid for directed graphs.")
        return sum(freq for (_, _, freq) in self.adjacency_dict[vertex])

    def in_degree(self, vertex):
        if not self.directed:
            raise ValueError("This method is only valid for directed graphs.")
        vertex_index = self.sorted_vertices.index(vertex)
        if isinstance(self.adjacency_matrix, np.ndarray):
            return np.sum(self.adjacency_matrix[:, vertex_index])
        else:
            return sum(row[vertex_index] for row in self.adjacency_matrix)

    def get_adjacency_matrix(self):
        """Say that this current version only applies for non-weighted matrices"""
        
        sorted_vertices_list = sorted(self.adjacency_dict.items())

        frequency_dictionaries = {}
        subdictionary = {}
        
        vertices_list_keys = list(self.adjacency_dict.keys())

        for key in sorted(vertices_list_keys):
            subdictionary[key] = 0
        for key in sorted(vertices_list_keys):
            frequency_dictionaries[key] = subdictionary.copy()
        
        for vertex_pairs in sorted_vertices_list:
            listed_vertices_connections = list(vertex_pairs[1])

            for i in range(len(listed_vertices_connections)):
                observed_vertex = listed_vertices_connections[i]
                for j in range(len(listed_vertices_connections)):
                    if i < j:
                        if observed_vertex[0] == listed_vertices_connections[j][0]:
                            frequency_dictionaries[vertex_pairs[0]][observed_vertex[0]] += listed_vertices_connections[j][2]
                frequency_dictionaries[vertex_pairs[0]][observed_vertex[0]] += observed_vertex[2]
        self.frequency_dictionaries = frequency_dictionaries

        adjacency_matrix = [
        [frequency_dictionaries[v[0]][u[0]] for u in sorted_vertices_list]
        for v in sorted_vertices_list]

        self.adjacency_matrix = adjacency_matrix
        return adjacency_matrix

    def print_matrix(self,matrix):
        keys = []
        print("0|",end="")
        max_number_length = len(str(np.max(matrix)))
        min_number_length = len(str(np.min(matrix)))

        largest_digit_length = max(max_number_length,min_number_length)

        for key in self.frequency_dictionaries.keys():
            keys.append(key)
            print(key + " "*largest_digit_length,end="")
        print('')
        dim = len(keys)
        for i in range(dim):
            print(keys[i]+"|",end="")
            for j in range(dim):
                digits = str(matrix[i][j])
                print(digits+" "*(largest_digit_length-len(digits)+1),end="")
            print()
        print("")

    def get_n_step_walks(self,n:int):

        if n < 1:
            raise ValueError("n must be a positive integer.")

        self.n_step_walk_matrix = np.linalg.matrix_power(self.adjacency_matrix, n)
        return self.n_step_walk_matrix
    
    def get_cumulative_n_step_walks(self,n:int):
        if n < 1:
            raise ValueError("n must be a positive integer.")

        cumulative_matrix = np.zeros_like(self.adjacency_matrix)
        for step in range(1, n + 1):
            cumulative_matrix += np.linalg.matrix_power(self.adjacency_matrix, step)
        self.cumulative_n_step_walk_matrix = cumulative_matrix
        return self.cumulative_n_step_walk_matrix
    
    def get_degree_matrix(self):
        
        degree_matrix = np.zeros((self.order,self.order),dtype=int)
        for i in range(self.order):
            degree_matrix[i,i] = int(self.degree(self.sorted_vertices[i]))

        self.degree_matrix = degree_matrix
        return degree_matrix
        
    def get_laplacian_matrix(self):
        if self.degree_matrix is None:
            self.get_degree_matrix()

        self.laplacian_matrix = self.degree_matrix - self.adjacency_matrix
        return self.laplacian_matrix
    
    def get_simple_undirected_copy(self):
        if self.directed == False and self.multigraph == False and self.weighted == False and self.pseudograph == False:
            self.simple_undirected_copy = copy.deepcopy(self)
        else:
            simple_undirected_copy = copy.deepcopy(self)

            simple_undirected_copy.pseudograph = False
            simple_undirected_copy.directed = False
            simple_undirected_copy.multigraph = False
            simple_undirected_copy.weighted = False

            simple_undirected_copy._process_edges()
            simple_undirected_copy.reset()

            self.simple_undirected_copy = simple_undirected_copy
    
    def check_connectivity(self):
        if self.directed == True or self.multigraph == True or self.pseudograph == True:
            if self.simple_undirected_copy == None:
                self.get_simple_undirected_copy()
            
            if self.simple_undirected_copy.laplacian_matrix is None:
                self.simple_undirected_copy.get_laplacian_matrix()
                
            eigenvalues = np.linalg.eigvals(self.simple_undirected_copy.laplacian_matrix)
            tolerance = 1e-9
            zero_multiplicity = np.sum(np.isclose(eigenvalues, 0, atol=tolerance))
            if zero_multiplicity == 1:
                return True
            else:
                return False

        else:
            if self.laplacian_matrix is None:
                self.get_laplacian_matrix()
            eigenvalues = np.linalg.eigvals(self.laplacian_matrix)
            tolerance = 1e-9
            zero_multiplicity = np.sum(np.isclose(eigenvalues, 0, atol=tolerance))
            if zero_multiplicity == 1:
                return True
            else:
                return False

    def get_all(self):
        self.sorted_vertices = sorted(list(self.vertices))
        self.get_adjacency_dict()
        self.get_adjacency_matrix()
        self.get_degree_matrix()
        self.get_laplacian_matrix()
        self.get_simple_undirected_copy()

    def __repr__(self):
        edges_str = ", ".join(repr(e) for e in self.edges)
        return (
            f"Name: {self.name}\n\n"
            f"Order: {self.order}\n"
            f"Size: {self.size}\n\n"
            f"Vertices: {self.vertices}\n"
            f"Edges: {{{edges_str}}}\n\n"
            f"Directed: {self.directed}\n"
            f"Multigraph: {self.multigraph}\n"
            f"Weighted: {self.weighted}\n"
            f"Pseudograph: {self.pseudograph}\n\n"
        )