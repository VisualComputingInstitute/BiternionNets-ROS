/* Based on J. Leffer's version at http://stackoverflow.com/a/675193/2366315 */
#include <errno.h>
#include <string.h>
#include <memory.h>
#include <stdlib.h>
#include <sys/stat.h>

static int do_mkdir(const char *path, mode_t mode)
{
    struct stat st;

    if(stat(path, &st) != 0) {
        /* Directory does not exist. EEXIST for race condition */
        if(mkdir(path, mode) != 0 && errno != EEXIST) {
            return -1;
        }
    } else if(!S_ISDIR(st.st_mode)) {
        errno = ENOTDIR;
        return -1;
    }
    return 0;
}

/*
** mkpath - ensure all directories in path exist
** Algorithm takes the pessimistic view and works top-down to ensure
** each directory in path exists, rather than optimistically creating
** the last element and working backwards.
*/
int mkpath2(const char *path, mode_t mode)
{
    char *sp;
    char *copypath = strdup(path);

    int status = 0;
    char *pp = copypath;
    while (status == 0 && (sp = strchr(pp, '/')) != 0) {
        if (sp != pp) {
            /* Neither root nor double slash in path */
            *sp = '\0';
            status = do_mkdir(copypath, mode);
            *sp = '/';
        }
        pp = sp + 1;
    }
    if (status == 0)
        status = do_mkdir(path, mode);

    free(copypath);
    return status;
}

int mkpath(const char *path)
{
    return mkpath2(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}
