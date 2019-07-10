/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

#define LAPLACE_OPERATOR "laplace"
#define GELMGOLS_OPERATOR "gelmgols"

static const double EPS = 1.e-8;

static inline _Bool isEq(const double a, const double b)
{
    return -(a - b) < EPS && (a - b) < EPS;
}

static inline _Bool isGeq(const double a, const double b)
{
    return a > b || isEq(a, b);
}

static inline _Bool isLeq(const double a, const double b)
{
    return a < b || isEq(a, b);
}

static PyObject *OperatorNameError;


static PyObject* line_segment(PyObject* self, PyObject* args, PyObject* keywds)
{
    double a, b, kappa;
    char *operator_type;
    PyArrayObject *in_array_x, *in_array_s;
    PyObject      *out_array;
    NpyIter *in_iter_x, *in_iter_s;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext_x, *in_iternext_s;
    NpyIter_IterNextFunc *out_iternext;

    static char *kwlist[] = {"x", "s", "ab", "kappa", "operator", NULL};

    /*  parse single numpy array argument */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!(dd)|ds", kwlist, &PyArray_Type, &in_array_x, &PyArray_Type, &in_array_s, &a, &b, &kappa, &operator_type))
        return NULL;

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array_x, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter_x = NpyIter_New(in_array_x, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_s = NpyIter_New(in_array_s, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (in_iter_x == NULL || in_iter_s == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter_x);
        NpyIter_Deallocate(in_iter_s);
        goto fail;
    }

    in_iternext_x = NpyIter_GetIterNext(in_iter_x, NULL);
    in_iternext_s = NpyIter_GetIterNext(in_iter_s, NULL);
    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext_x == NULL || in_iternext_s == NULL || out_iternext == NULL) {
        NpyIter_Deallocate(in_iter_x);
        NpyIter_Deallocate(in_iter_s);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr_x = (double **) NpyIter_GetDataPtrArray(in_iter_x);
    double ** in_dataptr_s = (double **) NpyIter_GetDataPtrArray(in_iter_s);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    /*  iterate over the arrays */
    if(strcmp(operator_type, LAPLACE_OPERATOR) == 0)
    {
        do {
            if(isGeq(**in_dataptr_x, a) && isLeq(**in_dataptr_x, **in_dataptr_s))
                **out_dataptr = ((**in_dataptr_x - a) * (b - **in_dataptr_s)) / (b - a);
            else if(**in_dataptr_x > **in_dataptr_s && isLeq(**in_dataptr_x, b))
                **out_dataptr = ((**in_dataptr_s - a) * (b - **in_dataptr_x)) / (b - a);
            else
                **out_dataptr = 0.0;
        } while(in_iternext_x(in_iter_x) && in_iternext_s(in_iter_s) && out_iternext(out_iter));
    }
    else if (strcmp(operator_type, GELMGOLS_OPERATOR) == 0)
    {
        do {
            if(isGeq(**in_dataptr_x, a) && isLeq(**in_dataptr_x, **in_dataptr_s))
                **out_dataptr = (sinh(kappa * (**in_dataptr_x - a)) * sinh(kappa * (b - **in_dataptr_s))) / (kappa * sinh(kappa * (b - a)));
            else if(**in_dataptr_x > **in_dataptr_s && isLeq(**in_dataptr_x, b))
                **out_dataptr = (sinh(kappa * (**in_dataptr_s - a)) * sinh(kappa * (b - **in_dataptr_x))) / (kappa * sinh(kappa * (b - a)));
            else
                **out_dataptr = 0.0;
        } while(in_iternext_x(in_iter_x) && in_iternext_s(in_iter_s) && out_iternext(out_iter));
    }
    else
    {
        PyErr_SetString(OperatorNameError, "Wrong name of operator. Available operators: 'laplace', 'gelmgols'.");
        return NULL;
    }


    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter_x);
    NpyIter_Deallocate(in_iter_s);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}



static PyObject* square(PyObject* self, PyObject* args, PyObject* keywds)
{
    int n, m;
    double a, b, kappa;
    char *operator_type;
    PyArrayObject *in_array_x1, *in_array_x2, *in_array_s1, *in_array_s2;
    PyObject      *out_array;
    NpyIter *in_iter_x1, *in_iter_x2, *in_iter_s1, *in_iter_s2;
    NpyIter *out_iter;
    NpyIter_IterNextFunc *in_iternext_x1, *in_iternext_x2, *in_iternext_s1, *in_iternext_s2;
    NpyIter_IterNextFunc *out_iternext;

    /*  parse single numpy array argument */
    static char *kwlist[] = {"x1", "x2", "s1", "s2", "ab", "n", "kappa", "operator", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!(dd)i|ds", kwlist, \
                          &PyArray_Type, &in_array_x1, &PyArray_Type, &in_array_x2, \
                          &PyArray_Type, &in_array_s1, &PyArray_Type, &in_array_s2, \
                          &a, &b, &n, &kappa, &operator_type))
        return NULL;

    m = n;

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array_x1, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    /*  create the iterators */
    in_iter_x1 = NpyIter_New(in_array_x1, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_x2 = NpyIter_New(in_array_x2, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_s1 = NpyIter_New(in_array_s1, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    in_iter_s2 = NpyIter_New(in_array_s2, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (in_iter_x1 == NULL || in_iter_x2 == NULL || in_iter_s1 == NULL || in_iter_s2 == NULL)
        goto fail;

    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (out_iter == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
        NpyIter_Deallocate(in_iter_s1);
        NpyIter_Deallocate(in_iter_s2);
        goto fail;
    }

    in_iternext_x1 = NpyIter_GetIterNext(in_iter_x1, NULL);
    in_iternext_x2 = NpyIter_GetIterNext(in_iter_x2, NULL);
    in_iternext_s1 = NpyIter_GetIterNext(in_iter_s1, NULL);
    in_iternext_s2 = NpyIter_GetIterNext(in_iter_s2, NULL);

    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
    if (in_iternext_x1 == NULL || in_iternext_x2 == NULL || \
        in_iternext_s1 == NULL || in_iternext_s2 == NULL || \
        out_iternext == NULL) {
        NpyIter_Deallocate(in_iter_x1);
        NpyIter_Deallocate(in_iter_x2);
        NpyIter_Deallocate(in_iter_s1);
        NpyIter_Deallocate(in_iter_s2);
        NpyIter_Deallocate(out_iter);
        goto fail;
    }
    double ** in_dataptr_x1 = (double **) NpyIter_GetDataPtrArray(in_iter_x1);
    double ** in_dataptr_x2 = (double **) NpyIter_GetDataPtrArray(in_iter_x2);
    double ** in_dataptr_s1 = (double **) NpyIter_GetDataPtrArray(in_iter_s1);
    double ** in_dataptr_s2 = (double **) NpyIter_GetDataPtrArray(in_iter_s2);
    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);

    /*  iterate over the arrays */
    register double sum = 0.0;
    register double pni, qmi;
    if(strcmp(operator_type, LAPLACE_OPERATOR) == 0)
    {
        do {
            sum = 0.0;
            for(int ni = 1; ni <= n; ++ni)
            {
                pni = (M_PI * ni) / a;
                for(int mi = 1; mi <= m; ++mi)
                {
                    qmi = (M_PI * mi) / b;
                    sum += (sin(pni * (**in_dataptr_x1)) * sin(qmi * (**in_dataptr_x2)) * \
                            sin(pni * (**in_dataptr_s1)) * sin(qmi * (**in_dataptr_s2))) / \
                            (pni * pni + qmi * qmi);
                }
            }
            **out_dataptr = (4. / (a * b)) * sum;

        } while(in_iternext_x1(in_iter_x1) && in_iternext_x2(in_iter_x2) && \
                in_iternext_s1(in_iter_s1) && in_iternext_s2(in_iter_s2) && \
                out_iternext(out_iter));
    }
    else if(strcmp(operator_type, GELMGOLS_OPERATOR) == 0)
    {
        do {
            sum = 0.0;
            for(int ni = 1; ni <= n; ++ni)
            {
                pni = (M_PI * ni) / a;
                for(int mi = 1; mi <= m; ++mi)
                {
                    qmi = (M_PI * mi) / b;
                    sum += (sin(pni * (**in_dataptr_x1)) * sin(qmi * (**in_dataptr_x2)) * \
                            sin(pni * (**in_dataptr_s1)) * sin(qmi * (**in_dataptr_s2))) / \
                            (pni * pni + qmi * qmi + kappa * kappa);
                }
            }
            **out_dataptr = (4. / (a * b)) * sum;

        } while(in_iternext_x1(in_iter_x1) && in_iternext_x2(in_iter_x2) && \
                in_iternext_s1(in_iter_s1) && in_iternext_s2(in_iter_s2) && \
                out_iternext(out_iter));
    }
    else
    {
        PyErr_SetString(OperatorNameError, "Wrong name of operator. Available operators: 'laplace', 'gelmgols'.");
        return NULL;
    }


    /*  clean up and return the result */
    NpyIter_Deallocate(in_iter_x1);
    NpyIter_Deallocate(in_iter_x2);
    NpyIter_Deallocate(in_iter_s1);
    NpyIter_Deallocate(in_iter_s2);
    NpyIter_Deallocate(out_iter);
    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}



//static PyObject* circle(PyObject* self, PyObject* args, PyObject* keywds)
//{
//    int n, m;
//    double R;
//    char *operator_type;
//    PyArrayObject *in_array_r, *in_array_phi, *in_array_rho, *in_array_psi;
//    PyObject      *out_array;
//    NpyIter *in_iter_r, *in_iter_phi, *in_iter_rho, *in_iter_psi;
//    NpyIter *out_iter;
//    NpyIter_IterNextFunc *in_iternext_r, *in_iternext_phi, *in_iternext_rho, *in_iternext_psi;
//    NpyIter_IterNextFunc *out_iternext;
//
//    /*  parse single numpy array argument */
//    static char *kwlist[] = {"r", "phi", "rho", "psi", "R", "n", "operator", NULL};
//
//    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O!O!O!O!d|ds", kwlist, \
//                          &PyArray_Type, &in_array_r, &PyArray_Type, &in_array_phi, \
//                          &PyArray_Type, &in_array_rho, &PyArray_Type, &in_array_psi, \
//                          &R, &n, &operator_type))
//        return NULL;
//
//    m = n;
//
//    /*  construct the output array, like the input array */
//    out_array = PyArray_NewLikeArray(in_array_r, NPY_ANYORDER, NULL, 0);
//    if (out_array == NULL)
//        return NULL;
//
//    /*  create the iterators */
//    in_iter_r = NpyIter_New(in_array_r, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    in_iter_phi = NpyIter_New(in_array_phi, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    in_iter_rho = NpyIter_New(in_array_rho, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    in_iter_psi = NpyIter_New(in_array_psi, NPY_ITER_READONLY, NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    if (in_iter_r == NULL || in_iter_phi == NULL || in_iter_rho == NULL || in_iter_psi == NULL)
//        goto fail;
//
//    out_iter = NpyIter_New((PyArrayObject *)out_array, NPY_ITER_READWRITE,
//                          NPY_KEEPORDER, NPY_NO_CASTING, NULL);
//    if (out_iter == NULL) {
//        NpyIter_Deallocate(in_iter_r);
//        NpyIter_Deallocate(in_iter_phi);
//        NpyIter_Deallocate(in_iter_rho);
//        NpyIter_Deallocate(in_iter_psi);
//        goto fail;
//    }
//
//    in_iternext_r = NpyIter_GetIterNext(in_iter_r, NULL);
//    in_iternext_phi = NpyIter_GetIterNext(in_iter_phi, NULL);
//    in_iternext_rho = NpyIter_GetIterNext(in_iter_rho, NULL);
//    in_iternext_psi = NpyIter_GetIterNext(in_iter_psi, NULL);
//
//    out_iternext = NpyIter_GetIterNext(out_iter, NULL);
//    if (in_iternext_r == NULL || in_iternext_phi == NULL || \
//        in_iternext_rho == NULL || in_iternext_psi == NULL || \
//        out_iternext == NULL) {
//        NpyIter_Deallocate(in_iter_r);
//        NpyIter_Deallocate(in_iter_phi);
//        NpyIter_Deallocate(in_iter_rho);
//        NpyIter_Deallocate(in_iter_psi);
//        NpyIter_Deallocate(out_iter);
//        goto fail;
//    }
//    double ** r = (double **) NpyIter_GetDataPtrArray(in_iter_r);
//    double ** phi = (double **) NpyIter_GetDataPtrArray(in_iter_phi);
//    double ** rho = (double **) NpyIter_GetDataPtrArray(in_iter_rho);
//    double ** psi = (double **) NpyIter_GetDataPtrArray(in_iter_psi);
//    double ** out_dataptr = (double **) NpyIter_GetDataPtrArray(out_iter);
//
//    /*  iterate over the arrays */
//    if(strcmp(operator_type, LAPLACE_OPERATOR) == 0)
//    {
//        do {
//            **out_dataptr = (1. / (4. * M_PI)) * \
//               log((**r * **r * **rho * **rho - 2 * R * R * **r * **rho * cos(**phi - **psi) + R * R * R * R) / \
//               (R * R * (**r * **r - 2 * **r * **rho * cos(**phi - **psi) + **rho * **rho)));
//
//        } while(in_iternext_r(in_iter_r) && in_iternext_phi(in_iter_phi) && \
//                in_iternext_rho(in_iter_rho) && in_iternext_psi(in_iter_psi) && \
//                out_iternext(out_iter));
//    }
    // TODO:
//    else if(strcmp(operator_type, GELMGOLS_OPERATOR) == 0)
//    {
//        register double sum;
//        do {
//            sum = 0.0;
//            for(int ni = 1; ni <= n; ++ni)
//            {
//                pni = (M_PI * ni) / a;
//                for(int mi = 1; mi <= m; ++mi)
//                {
//                    qmi = (M_PI * mi) / b;
//                    sum += NULL;
//                }
//            }
//            **out_dataptr = (4. / (a * b)) * sum;
//
//        } while(in_iternext_x1(in_iter_x1) && in_iternext_x2(in_iter_x2) && \
//                in_iternext_s1(in_iter_s1) && in_iternext_s2(in_iter_s2) && \
//                out_iternext(out_iter));
//    }
//    else
//    {
//        PyErr_SetString(OperatorNameError, "Wrong name of operator. Available operators: 'laplace', 'gelmgols'.");
//        return NULL;
//    }
//
//
//    /*  clean up and return the result */
//    NpyIter_Deallocate(in_iter_r);
//    NpyIter_Deallocate(in_iter_phi);
//    NpyIter_Deallocate(in_iter_rho);
//    NpyIter_Deallocate(in_iter_psi);
//    NpyIter_Deallocate(out_iter);
//    Py_INCREF(out_array);
//    return out_array;
//
//    /*  in case bad things happen */
//    fail:
//        Py_XDECREF(out_array);
//        return NULL;
//}


/*  define functions in module */
static PyMethodDef GreenMethods[] =
{
     {"line_segment", (PyCFunction)line_segment, METH_VARARGS|METH_KEYWORDS, ""},
     {"square", (PyCFunction)square, METH_VARARGS|METH_KEYWORDS, ""},
     //{"circle", (PyCFunction)circle, METH_VARARGS|METH_KEYWORDS, ""},
     {NULL, NULL, 0, NULL}
};

static struct PyModuleDef greenmodule = {
    PyModuleDef_HEAD_INIT,
    "green",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    GreenMethods
};

PyMODINIT_FUNC
PyInit_green(void)
{
    import_array();
    PyObject *m;
    m = PyModule_Create(&greenmodule);
    if (!m)
        return NULL;

    OperatorNameError = PyErr_NewException("operator_name.error", NULL, NULL);
    Py_INCREF(OperatorNameError);
    PyModule_AddObject(m, "error", OperatorNameError);

    return m;
}
