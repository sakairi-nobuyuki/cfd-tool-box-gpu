class FieldVars1D {
    protected:
        double *cArray, *gArray;
        int n_bytes;

        void initVarsWithZero();
    public:
        int n_len;
        char name[64];

        FieldVars1D(int array_length, char var_name[64]);
        ~FieldVars1D();

        void output(double time);
        
};