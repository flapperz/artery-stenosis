cd /opt/s
cmake \
  -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=14.0 \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  -DQt5_DIR:PATH=/Users/flap/Qt/5.15.2/clang_64/lib/cmake/Qt5 \
  /Users/flap/Source/Slicer

# then do make -j6sq