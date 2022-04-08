// Copyright (c) 2020 Mugilan Mariappan, Joanna Che and Keval Vora.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef __RWLOCK_H__
#define __RWLOCK_H__
#include "parallel.h"

class RWLock {
#ifndef MIND_MUTEX
  pthread_rwlock_t rwlock;

public:
  void init() { pthread_rwlock_init(&rwlock, NULL); }

  void readLock() { pthread_rwlock_rdlock(&rwlock); }

  void writeLock() { /*printf("try %p\n", &rwlock);*/ pthread_rwlock_wrlock(&rwlock); /*printf("get %p\n", &rwlock);*/}

  void unlock() { /*printf("try u %p\n", &rwlock);*/ pthread_rwlock_unlock(&rwlock); /*printf("u %p\n", &rwlock);*/}

  void destroy() { pthread_rwlock_destroy(&rwlock); }
#else
  pthread_mutex_t mutex;

public:
  void init() { pthread_mutex_init(&mutex, NULL); }

  void readLock() { pthread_mutex_lock(&mutex); }

  void writeLock() { pthread_mutex_lock(&mutex); }

  void unlock() { pthread_mutex_unlock(&mutex); }

  void destroy() { pthread_mutex_destroy(&mutex); }
#endif
};

#endif //__RWLOCK_HPP__
