-- Test vector operations
SELECT vector_add('[1.0, 2.0]'::vector, '[3.0, 4.0]'::vector);
SELECT vector_sub('[5.0, 7.0]'::vector, '[2.0, 3.0]'::vector);
SELECT vector_mul('[2.0, 3.0]'::vector, 2.5);

-- Test vector dimensions
SELECT vector_dims('[1.0]'::vector);
SELECT vector_dims('[1.0, 2.0]'::vector);
SELECT vector_dims('[1.0, 2.0, 3.0, 4.0, 5.0]'::vector);

-- Test array conversion
SELECT array_to_vector(ARRAY[1.0, 2.0, 3.0]::real[]);
SELECT vector_to_array('[1.0, 2.0, 3.0]'::vector);
