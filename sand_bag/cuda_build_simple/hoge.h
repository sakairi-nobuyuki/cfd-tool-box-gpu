#ifndef __HOGE__
#define __HOGE__

class Hoge{
    private:
        int _n_block_x, _n_thread_x, _n_size, _n_size_orig;
        int *_u_gpu, *_u_cpu;
    public:
        Hoge(int n_size);
        void sum();
        void print();


};


#endif