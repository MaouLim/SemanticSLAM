cmake_minimum_required(VERSION 2.8)
project(ALG)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

set(ALGLIB_HEADERS
include/alglibinternal.h
include/diffequations.h
include/linalg.h
include/statistics.h
include/alglibmisc.h
include/fasttransforms.h
include/optimization.h
include/stdafx.h
include/ap.h
include/integration.h
include/solvers.h
include/dataanalysis.h
include/interpolation.h
include/specialfunctions.h
)
set(ALGLIB_SRC
src/alglibinternal.cpp
src/diffequations.cpp
src/linalg.cpp            
src/statistics.cpp
src/alglibmisc.cpp      
src/fasttransforms.cpp  
src/optimization.cpp
src/ap.cpp              
src/integration.cpp     
src/solvers.cpp
src/dataanalysis.cpp    
src/interpolation.cpp   
src/specialfunctions.cpp
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
add_library(ALGLIB SHARED ${ALGLIB_SRC})

