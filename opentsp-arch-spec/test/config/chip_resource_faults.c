#define _GNU_SOURCE
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>
extern void *__libc_calloc(size_t, size_t);
extern pid_t __libc_fork(void);
extern void *__libc_realloc(void *, size_t);
static pid_t original_pid;
__attribute__((constructor)) static void origin(void) { original_pid = getpid(); }
static int allocation, forks, pipes;
static int selected(const char *name, int n) { const char *s = getenv(name); return s && atoi(s) == n; }
static void note(const char *kind, int n, long value) {
  char b[100]; int len = snprintf(b, sizeof b, "TEST %s %d %ld\n", kind, n, value);
  if (len > 0) write(2, b, (size_t)len);
}
void *calloc(size_t n, size_t size) {
  if (n == 2 && (size == 4 || size == 8 || size == 16)) {
    allocation++; note("allocation", allocation, size);
    if (selected("OPENTSP_TEST_ALLOC", allocation)) return NULL;
  }
  return __libc_calloc(n,size);
}
pid_t fork(void) {
  forks++;
  if (selected("OPENTSP_TEST_FORK",forks)) { errno=EAGAIN; note("fork-failed",forks,-1); return -1; }
  pid_t p=__libc_fork(); if(p>0)note("forked",forks,p); return p;
}
int pipe(int fds[2]) {
  pipes++;
  if(selected("OPENTSP_TEST_PIPE",pipes)){ errno=EMFILE; note("pipe-failed",pipes,-1); return -1; }
  return syscall(SYS_pipe2,fds,0);
}

void *realloc(void *p, size_t size) {
  if (getpid() == original_pid && forks > 0 && getenv("OPENTSP_TEST_REALLOC")) { note("realloc-failed",1,size); return NULL; }
  return __libc_realloc(p,size);
}
ssize_t read(int fd, void *buffer, size_t size) {
  if (getpid() == original_pid && forks > 0 && getenv("OPENTSP_TEST_READ")) { errno=EIO; note("read-failed",1,fd); return -1; }
  return syscall(SYS_read,fd,buffer,size);
}
ssize_t write(int fd, const void *buffer, size_t size) {
  if (fd != 2 && getpid() == original_pid && forks > 0 && getenv("OPENTSP_TEST_EPIPE")) {
    int dead[2]; if (syscall(SYS_pipe2,dead,0)) _exit(90);
    close(dead[0]); if (dup2(dead[1],fd)<0) _exit(91); close(dead[1]);
    note("epipe-write",1,fd); return syscall(SYS_write,fd,buffer,size);
  }
  if (fd != 2 && getpid() == original_pid && forks > 0 && getenv("OPENTSP_TEST_WRITE")) { errno=EIO; note("write-failed",1,fd); return -1; }
  return syscall(SYS_write,fd,buffer,size);
}
