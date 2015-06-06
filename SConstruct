env = Environment(
        CPPPATH=["/usr/include/python2.7", '/usr/local/include/opencv', '/usr/include/opencv',
            '/usr/lib/python2.7/site-packages/numpy/core/include',
            '/usr/include/libswscale/',
            '/usr/include/libavcodec', '/usr/include/libavformat'],
        LIBPATH=["/usr/local/lib"],
        CXXFLAGS=["-D__STDC_CONSTANT_MACROS"],
    )
#env.Replace(CFLAGS=['-O2','-Wall','-ansi','-pedantic'])
env.SharedLibrary(target = 'libext', SHLIBPREFIX="",
    LIBS=['opencv_highgui', 'opencv_core', 'avformat', 'avcodec' , 'swscale'],
    # source = ['ext.cpp', 'pelco.c'])
    source = ['ext.cpp'])
