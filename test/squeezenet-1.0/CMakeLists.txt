# Squeezenet is only used in the performance test bench.
# Squeeznet with three different input datasets.
# The error (5) must be big for execution to pass since either:
# there is a error in the generated code or rounding errors accumulate.
# Also, squeezenet model does not terminate with a sigmoid, so output
# values are not in the range [0,1]
add_perftest( squeezenet-1.0 ${CMAKE_CURRENT_SOURCE_DIR} 5 0 )
add_perftest( squeezenet-1.0 ${CMAKE_CURRENT_SOURCE_DIR} 5 1 )
add_perftest( squeezenet-1.0 ${CMAKE_CURRENT_SOURCE_DIR} 5 2 )

# propagate contents of PERFTESTS from add_perftest() to out parent scope.
set( PERFTESTS ${PERFTESTS} PARENT_SCOPE )
set( PERFRUNS ${PERFRUNS} PARENT_SCOPE )

