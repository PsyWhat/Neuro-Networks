using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;


//using IntPtr = IntPtr;

namespace MathOps
{
    [StructLayout(LayoutKind.Sequential)]
    struct MatUMS
    {
        public int rows;
        public int cols;
        public IntPtr values;
    }

    static class MatrixOpsWrapper
    {
        /*
         	EXPORT IntPtr MatrixGetNew( int rows , int columns );

	        EXPORT void MatrixNull( Matrix *a );

	        EXPORT IntPtr MatrixCopy( Matrix *a );

	        EXPORT IntPtr MatrixTransposed( Matrix *a , int numThreads );

	        EXPORT void FreeMatrix( Matrix *m );

	        EXPORT double MatrixGetElem( Matrix *a , int row , int column );

	        EXPORT void MatrixSetElem( Matrix *a , int row , int column , double value );

	        EXPORT IntPtr MatrixColConcat( Matrix *a , Matrix *b );

	        EXPORT IntPtr MatrixRowConcat( Matrix *a , Matrix *b );

	        EXPORT IntPtr MatrixMultiply( Matrix *a , Matrix *b , int maxThreads = 16 );

	        EXPORT IntPtr MatrixMultiplyDouble( Matrix *a , double d , int maxThreads = 16 );

	        EXPORT IntPtr MatrixAddition( Matrix *a , Matrix *b , int maxThreads = 16 );

	        EXPORT IntPtr MatrixHadamardProduct( Matrix *a , Matrix *b , int maxThreads = 16 );
        */


        [DllImport("kernel32", CharSet = CharSet.Auto, SetLastError = true)]
        static extern IntPtr LoadLibrary(string dllToLoad);

        [DllImport("kernel32", CharSet = CharSet.Ansi, SetLastError = true)]
        static extern IntPtr GetProcAddress(IntPtr hModule, string procedureName);

        [DllImport("kernel32", CharSet = CharSet.Auto)]
        static extern ushort GetLastError();

        [DllImport("kernel32", CharSet = CharSet.Auto)]
        static extern bool FreeLibrary(IntPtr hModule);


        public delegate IntPtr MatrixGetNewDelegate(int rows, int columns);
        public delegate void MatrixNullDelegate(IntPtr mat);
        public delegate IntPtr MatrixCopyDelegate(IntPtr mat);
        public delegate IntPtr MatrixTransposedDelegate(IntPtr mat, int maxThreads = 16);
        public delegate void FreeMatrixDelegate(IntPtr mat);
        public delegate double MatrixGetElemDelegate(IntPtr a, int row, int column);
        public delegate void MatrixSetElemDelegate(IntPtr a, int row, int column, double value);
        public delegate IntPtr MatrixColConcatDelegate(IntPtr a, IntPtr b);
        public delegate IntPtr MatrixRowConcatDelegate(IntPtr a, IntPtr b);
        public delegate IntPtr MatrixMultiplyDelegate(IntPtr a, IntPtr b, int maxThreads = 16);
        public delegate IntPtr MatrixMultiplyDoubleDelegate(IntPtr a, double d, int maxThreads = 16);
        public delegate IntPtr MatrixAdditionDelegate(IntPtr a, IntPtr b, int maxThreads = 16);
        public delegate IntPtr MatrixHadamardProductDelegate(IntPtr a, IntPtr b, int maxThreads = 16);

        public delegate IntPtr MatrixGetRowDelegate(IntPtr a, int row);
        public delegate IntPtr MatrixGetColumnDelegate(IntPtr a, int row);

        public static MatrixGetNewDelegate New;
        public static MatrixNullDelegate FillWithNulls;
        public static MatrixCopyDelegate Copy;
        public static MatrixTransposedDelegate Transposed;
        public static FreeMatrixDelegate Free;
        public static MatrixGetElemDelegate GetElem;
        public static MatrixSetElemDelegate SetElem;
        public static MatrixColConcatDelegate ColConcat;
        public static MatrixRowConcatDelegate RowConcat;
        public static MatrixMultiplyDelegate Multiply;
        public static MatrixMultiplyDoubleDelegate MultiplyDouble;
        public static MatrixAdditionDelegate Sum;
        public static MatrixHadamardProductDelegate Hadamard;
        public static MatrixGetRowDelegate GetRow;
        public static MatrixGetColumnDelegate GetColumn;

        public static int GetMatRows(IntPtr m)
        {
            return Marshal.PtrToStructure<MatUMS>(m).rows;
        }
        public static int GetMatCols(IntPtr m)
        {
            return Marshal.PtrToStructure<MatUMS>(m).cols;
        }

        static MatrixOpsWrapper()
        {
            /*string pv = "x86";
            if (Environment.Is64BitProcess)
            {
                pv = "x64";
            }
            string filePath = string.Format("{0}\\{1}\\{2}", Environment.CurrentDirectory, DllPath, string.Format(DllCont, pv));

            IntPtr libPtr = LoadLibrary(filePath);

            if (libPtr == IntPtr.Zero)
            {
                throw new PlatformNotSupportedException("Can't load CNeuroNet DDL");
            }
            LibPTR = libPtr;

            var err = GetLastError();




            IntPtr fun = GetProcAddress(LibPTR, "AddConnection");

            err = GetLastError();

            __AddConnection =
                Marshal.GetDelegateForFunctionPointer<AddConnectionDelegate>(fun);*/

            const string dllName = "MatrixMults";
            const string dllFolder = "DLLs";

            string pv = "x86";

            if(Environment.Is64BitProcess)
            {
                pv = "x64";
            }
            string filePath = string.Format("{0}\\{1}\\{2}", Environment.CurrentDirectory, dllFolder, string.Concat(dllName, pv,".dll"));

            libPtr = LoadLibrary(filePath);

            if (libPtr == IntPtr.Zero)
            {
                throw new PlatformNotSupportedException("Can't load CNeuroNet DDL");
            }

            New = Marshal.GetDelegateForFunctionPointer<MatrixGetNewDelegate>(GetProcAddress(libPtr, "MatrixGetNew"));

            New(3, 5);

            FillWithNulls = Marshal.GetDelegateForFunctionPointer<MatrixNullDelegate>(GetProcAddress(libPtr, "MatrixNull"));

            Copy = Marshal.GetDelegateForFunctionPointer<MatrixCopyDelegate>(GetProcAddress(libPtr, "MatrixCopy"));

            Transposed = Marshal.GetDelegateForFunctionPointer<MatrixTransposedDelegate>(GetProcAddress(libPtr, "MatrixTransposed"));

            Free = Marshal.GetDelegateForFunctionPointer<FreeMatrixDelegate>(GetProcAddress(libPtr, "FreeMatrix"));

            GetElem = Marshal.GetDelegateForFunctionPointer<MatrixGetElemDelegate>(GetProcAddress(libPtr, "MatrixGetElem"));

            SetElem = Marshal.GetDelegateForFunctionPointer<MatrixSetElemDelegate>(GetProcAddress(libPtr, "MatrixSetElem"));


            ColConcat = Marshal.GetDelegateForFunctionPointer<MatrixColConcatDelegate>(GetProcAddress(libPtr, "MatrixColConcat"));

            RowConcat = Marshal.GetDelegateForFunctionPointer<MatrixRowConcatDelegate>(GetProcAddress(libPtr, "MatrixRowConcat"));

            Multiply = Marshal.GetDelegateForFunctionPointer<MatrixMultiplyDelegate>(GetProcAddress(libPtr, "MatrixMultiply"));

            MultiplyDouble = Marshal.GetDelegateForFunctionPointer<MatrixMultiplyDoubleDelegate>(GetProcAddress(libPtr, "MatrixMultiplyDouble"));

            Sum = Marshal.GetDelegateForFunctionPointer<MatrixAdditionDelegate>(GetProcAddress(libPtr, "MatrixAddition"));

            Hadamard = Marshal.GetDelegateForFunctionPointer<MatrixHadamardProductDelegate>(GetProcAddress(libPtr, "MatrixHadamardProduct"));

            GetRow = Marshal.GetDelegateForFunctionPointer<MatrixGetRowDelegate>(GetProcAddress(libPtr, "MatrixGetRow"));

            GetColumn = Marshal.GetDelegateForFunctionPointer<MatrixGetColumnDelegate>(GetProcAddress(libPtr, "MatrixGetColumn"));

        }

        static IntPtr libPtr;


    }
}
