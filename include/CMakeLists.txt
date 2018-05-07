target_include_directories(${PROJECT_NAME} PRIVATE .)

file(GLOB HEADER_FILES "*.h")
target_sources(${PROJECT_NAME} PUBLIC ${HEADER_FILES})
