using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathOps;

namespace LSTM
{
    public class LSTMLayer
    {
        MatrixD weightsForgetGate;
        MatrixD weightsIGate;
        MatrixD weightsCGate;
        MatrixD weightsOutputGate;

        VectorDColumn bF;
        VectorDColumn bI;
        VectorDColumn bC;
        VectorDColumn bO;
        VectorDColumn ctmo;

        VectorDColumn htmo;


        MatrixD tanActivationFunction(MatrixD vec)
        {
            throw new NotImplementedException();
        }
        MatrixD sigmoidActivationFunction(MatrixD vec)
        {
            throw new NotImplementedException();
        }

        public VectorDColumn Process(VectorDColumn data)
        {
            var ft = sigmoidActivationFunction( weightsForgetGate * VectorDColumn.Concatenation(htmo,data) +  bF);
            var it = sigmoidActivationFunction(weightsIGate * VectorDColumn.Concatenation(htmo, data) + bI);
            var Cst = tanActivationFunction(weightsCGate * VectorDColumn.Concatenation(htmo, data) + bC);
            var Ct = ft & ctmo + it & Cst;
            var ot = sigmoidActivationFunction( weightsOutputGate * VectorDColumn.Concatenation(htmo,data) + bO );
            var ht = ot & tanActivationFunction(Ct);

            htmo = ht.GetColumn();
            ctmo = Ct.GetColumn();


            return ht.GetColumn();
        }


    }
}
