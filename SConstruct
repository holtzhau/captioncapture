env = Environment(
        CPPPATH=["/usr/include/python2.7", '/usr/local/include/opencv', '/usr/include/opencv',
            '/usr/lib/python2.7/site-packages/numpy/core/include'],
        LIBPATH=["/usr/local/lib"],
        CXXFLAGS=["-D__STDC_CONSTANT_MACROS"],
    )
env.SharedLibrary(target = 'libext', SHLIBPREFIX="",
    LIBS=['opencv_highgui', 'opencv_core'],
    source = ['ext.cpp'])
