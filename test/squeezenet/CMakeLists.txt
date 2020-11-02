# Squeezenet is only used in the performance test bench.
# TODO: move these functions up in the hierarcy if there are more per tests.
function( add_perftest node_name dir accuracy input_set )

	ONNX_type_test_build(${node_name} ${dir} ${accuracy} ${input_set})
	set_property(
		TARGET ${node_name}_${input_set}_test
		PROPERTY EXCLUDE_FROM_ALL TRUE
	)
	set( PERFTESTS ${PERFTESTS};${node_name}_${input_set}_test PARENT_SCOPE)

	add_custom_target(
		${node_name}_${input_set}_test_run
		COMMAND time --format=%U ./${node_name}_${input_set}_test
		COMMENT "Running ${node_name} ${input_set}"
		DEPENDS perftest_build
	)
	set( PERFRUNS ${PERFRUNS};${node_name}_${input_set}_test_run PARENT_SCOPE)

endfunction()

# Squeeznet with three different input datasets.
# The error (5) must be big for execution to pass since either:
# there is a error in the generated code or rounding errors accumulate.
# Also, squeezenet model does not terminate with a sigmoid, so output
# values are not in the range [0,1]
add_perftest( squeezenet ${CMAKE_CURRENT_SOURCE_DIR} 5 0 )
add_perftest( squeezenet ${CMAKE_CURRENT_SOURCE_DIR} 5 1 )
add_perftest( squeezenet ${CMAKE_CURRENT_SOURCE_DIR} 5 2 )

add_custom_target(
	perftest_build
	DEPENDS ${PERFTESTS}
)
add_custom_target(
	perftest_run
	DEPENDS ${PERFRUNS}
)

