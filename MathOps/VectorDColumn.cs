using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MathOps
{
    public class VectorDColumn : MatrixD
    {

        public int Length
        {
            get { return this._rows; }
        }

        public VectorDColumn(int len):base(len,1)
        {

        }

        public VectorDColumn(VectorDColumn copy):base(copy)
        {
            
        }

        public VectorDColumn(MatrixD mat):base(mat)
        {
            if (this._columns != 1)
            {
                throw new ArgumentException("Bad matrix size.");
            }
        }

        protected VectorDColumn(IntPtr mat):base(mat)
        {
            if(this._columns != 1)
            {
                throw new ArgumentException("Bad matrix size.");
            }
        }
        

        public double this[int i]
        {
            get
            {
                if(i >= 0 && i < Length)
                {
                    return this[i, 1];
                }
                else
                {
                    throw new IndexOutOfRangeException("Indexing VectorDColumn out of range.");
                }
            }
            set
            {
                if( i >= 0 && i < Length)
                {
                    this[i, 1] = value;
                }
                else
                {
                    throw new IndexOutOfRangeException("Indexing VectorDColumn out of range.");
                }
            }
        }

        public VectorDRow Transposed()
        {
            return new VectorDRow(base.Transposed());
        }

        public static VectorDColumn Concatenation(VectorDColumn a, VectorDColumn b)
        {
            VectorDColumn res = new VectorDColumn(MatrixOpsWrapper.ColConcat(a._unmanagedMatrix, b._unmanagedMatrix));
            return res;
        }

    }
}
