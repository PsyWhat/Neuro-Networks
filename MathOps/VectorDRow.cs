using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MathOps
{
    public class VectorDRow : MatrixD
    {



        public int Length
        {
            get { return this._columns; }
        }

        public VectorDRow(int len):base(1,len)
        {
        }

        public VectorDRow(VectorDRow copy):base(copy)
        {
        }

        public VectorDRow(MatrixD copy):base(copy)
        {
            if(this._rows != 1)
            {
                throw new ArgumentException("Row matrix should have only 1 row");
            }
        }

        public VectorDRow(IntPtr mat):base(mat)
        {
            if (this._rows != 1)
            {
                throw new ArgumentException("Row matrix should have only 1 row");
            }
        }
        
        public VectorDColumn Transposed()
        {
            return new VectorDColumn(base.Transposed());
        }

        public static VectorDRow Concatenation(VectorDRow a, VectorDRow b)
        {
            VectorDRow res = new VectorDRow(MatrixOpsWrapper.RowConcat(a._unmanagedMatrix, b._unmanagedMatrix));
            return res;
        }
        

    }
}
