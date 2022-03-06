for lib in $(find /usr/lib/x86_64-linux-gnu/ -name \*.a) ; do echo $lib ; nm $lib | grep dlopen | grep -v " U "   ; done
