add_custom_target(
	perftest_build
	DEPENDS ${PERFTESTS}
)

add_custom_target(
	perftest_run
	DEPENDS ${PERFRUNS}
)

