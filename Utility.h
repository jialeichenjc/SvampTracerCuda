#ifndef UTILITYH
#define UTILITYH


#include <stdlib.h>
#include <time.h>

namespace util {
    
    inline double gen_rand() {
        //srand(time(NULL));
        return (double)rand() / RAND_MAX;
    }
}

#endif // !UTILITYH