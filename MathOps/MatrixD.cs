using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace MathOps
{
    public class MatrixD
    {

        protected IntPtr _unmanagedMatrix;

        protected int _rows;
        protected int _columns;

        /// <summary>
        /// Number of matrix rows.
        /// </summary>
        public int Rows { get { return _rows; } }

        /// <summary>
        /// Number of matrix columns.
        /// </summary>
        public int Columns { get { return _columns; } }


        #region Constructors

        protected MatrixD()
        {
            _rows = 0;
            _columns = 0;
            _unmanagedMatrix = IntPtr.Zero;
        }
        public MatrixD(int rows, int columns)
        {
            if(rows > 0 && columns > 0)

            {
                _rows = rows;
                _columns = columns;
                _unmanagedMatrix = MatrixOpsWrapper.New(_rows, _columns);

            }else
            {
                throw new ArgumentException("Size of matrix should pe positive non null.");
            }
        }

        public MatrixD(MatrixD copy)
        {
            _rows = copy._rows;
            _columns = copy._columns;
            _unmanagedMatrix = MatrixOpsWrapper.Copy(copy._unmanagedMatrix);
        }

        protected MatrixD(IntPtr data)
        {
            _unmanagedMatrix = data;
            _rows = MatrixOpsWrapper.GetMatRows(data);
            _columns = MatrixOpsWrapper.GetMatCols(data);
        }

        ~MatrixD()
        {
            MatrixOpsWrapper.Free(_unmanagedMatrix);
        }

        #endregion



        #region Operations


        public VectorDRow GetRow(int column = 0)
        {
            return new VectorDRow(new MatrixD(MatrixOpsWrapper.GetRow(_unmanagedMatrix, column)));
        }
        public VectorDColumn GetColumn(int row = 0)
        {
            return new VectorDColumn(new MatrixD(MatrixOpsWrapper.GetColumn(_unmanagedMatrix, row)));
        }

        /// <summary>
        /// Transposing the matrix.
        /// </summary>
        public MatrixD Transposed()
        {
            return new MatrixD(MatrixOpsWrapper.Transposed(this._unmanagedMatrix));
        }



        #endregion


        #region Operators

        /// <summary>
        /// Multiplication of two matrixes
        /// </summary>
        /// <param name="a">First matrix</param>
        /// <param name="b">Second matrix</param>
        /// <returns>The result of multiplication.</returns>
        public static MatrixD operator *(MatrixD a, MatrixD b)
        {
            if (a._columns == b._rows)
            {
                return new MatrixD(MatrixOpsWrapper.Multiply(a._unmanagedMatrix, b._unmanagedMatrix));
            }
            else
            {
                throw new ArgumentException("Columns of matrix a should be the same with rows of matrix b");
            }
        }

        /// <summary>
        /// Addition of two matrixes
        /// </summary>
        /// <param name="a">First matrix</param>
        /// <param name="b">Second matrix</param>
        /// <returns>The result of addition.</returns>
        public static MatrixD operator +(MatrixD a, MatrixD b)
        {
            if (a._rows == b._rows && a._columns == b._columns)
            {
                return new MatrixD(MatrixOpsWrapper.Sum(a._unmanagedMatrix,b._unmanagedMatrix));
            }
            else
            {
                throw new ArgumentException("Size of matrix a must be the same with size of matrix b.");
            }
        }

        /// <summary>
        /// Hadamard product.
        /// </summary>
        /// <param name="a">First matrix</param>
        /// <param name="b">Second matrix</param>
        /// <returns>The result of Hadamard product.</returns>
        public static MatrixD operator &(MatrixD a, MatrixD b)
        {
            if (a._rows == b._rows && a._columns == b._columns)
            {
                return new MatrixD(MatrixOpsWrapper.Hadamard(a._unmanagedMatrix, b._unmanagedMatrix));
            }else
            {
                throw new ArgumentException("Size of matrix a must be the same with size of matrix b.");
            }
        }

        /// <summary>
        /// Multilication of matrix and double.
        /// </summary>
        /// <param name="m">Matrix to multiply.</param>
        /// <param name="d">Double, whinch multiplying all values.</param>
        /// <returns>Multiplication of matrix and double.</returns>
        public static MatrixD operator *(MatrixD m, double d)
        {
            return new MatrixD(MatrixOpsWrapper.MultiplyDouble(m._unmanagedMatrix, d));
        }

        /// <summary>
        /// Multilication of matrix and double.
        /// </summary>
        /// <param name="d">Double, whinch multiplying all values.</param>
        /// <param name="m">Matrix to multiply.</param>
        /// <returns>Multiplication of matrix and double.</returns>
        public static MatrixD operator *(double d, MatrixD m)
        {
            return new MatrixD(MatrixOpsWrapper.MultiplyDouble(m._unmanagedMatrix, d));
        }


        public double this[int row, int column]
        {
            get
            {
                if (row >= 0 && row < _rows && column >= 0 && column < _columns)
                {
                    return MatrixOpsWrapper.GetElem(_unmanagedMatrix, row, column);
                }
                else
                {
                    throw new IndexOutOfRangeException(string.Format("Trying to access matrix out of bounds.row:[{0}] column:[{1}], rows:[{2}] columns:[{3}]", row, column, _rows, _columns));
                }
            }
            set
            {
                if (row >= 0 && row < _rows && column >= 0 && column < _columns)
                {
                    MatrixOpsWrapper.SetElem(_unmanagedMatrix, row, column, value);
                }
                else
                {
                    throw new IndexOutOfRangeException(string.Format("Trying to access matrix out of bounds.row:[{0}] column:[{1}], rows:[{2}] columns:[{3}]", row, column, _rows, _columns));
                }
            }
        }


        #endregion


        public override string ToString()
        {
            return string.Format("Matrix:[{0};{1}]",_rows,_columns);
        }

        public static string GetString(MatrixD m)
        {
            string res = "";
            res += "[";
            for(int i = 0; i < m.Rows; ++ i)
            {
                res += "[";
                for(int j = 0; j < m.Columns; ++j)
                {
                    res += m[i, j].ToString("F3");
                    if (j != m.Columns - 1)
                    {
                        res += ",";
                    }

                }
                res += "]";
                if (i != m.Rows - 1)
                {
                    res += ",";
                }
            }
            res += "]";
            return res;
        }

        public static void WriteToStream(MatrixD m, Stream stream)
        {
            BinaryWriter wr = new BinaryWriter(stream, Encoding.UTF32, true);
            wr.Write(m.Rows);
            wr.Write(m.Columns);
            for(int i = 0; i < m.Rows;++i)
            {
                for(int j = 0; j < m.Columns; ++ j)
                {
                    wr.Write(m[i,j]);
                }
            }
        }

        public static MatrixD ReadFromStream(Stream stream)
        {
            BinaryReader rr = new BinaryReader(stream, Encoding.UTF32, true);
            int rows = rr.ReadInt32();
            int columns = rr.ReadInt32();
            MatrixD m = new MatrixD(rows, columns);

            for (int i = 0; i < m.Rows; ++i)
            {
                for (int j = 0; j < m.Columns; ++j)
                {
                    m[i, j] = rr.ReadDouble();
                }
            }

            return m;
        }


    }
}
