-- Test basic type creation and operations
CREATE EXTENSION neurondb;

-- Test vector type
SELECT '[1.0, 2.0, 3.0]'::vector;
SELECT '[1.5, 2.5, 3.5, 4.5]'::vector;
SELECT vector_dims('[1.0, 2.0, 3.0, 4.0, 5.0]'::vector);

-- Test vectorp (packed)
SELECT vectorp_in('[1.0, 2.0, 3.0]')::text;
SELECT vectorp_dims(vectorp_in('[1.0, 2.0, 3.0, 4.0]'));

-- Test vecmap (sparse)
SELECT vecmap_in('{dim:10, nnz:3, indices:[0,3,7], values:[1.5,2.3,0.8]}')::text;

-- Test vgraph
SELECT vgraph_in('{nodes:5, edges:[[0,1],[1,2],[2,3],[3,4]]}')::text;

-- Test rtext
SELECT rtext_in('sample text for retrieval')::text;
