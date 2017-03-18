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


        public static double tanhActivationFunction(double x)
        {
            return Math.Tanh(x);
        }
        public static double tanhDiffActivationFunction(double x)
        {
            double th = Math.Tanh(x);
            return 1.0 - (th * th);
        }

        public static double sigmoidActivationFunction(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        public static double sigmoidDiffActivationFunction(double x)
        {
            double sigm = sigmoidActivationFunction(x);
            return sigm * (1.0 - sigm);
        }
        

        public VectorDColumn Process(VectorDColumn data)
        {
            var ft =  (weightsForgetGate * VectorDColumn.Concatenation(htmo,data) +  bF); ft.ApplyFunction(sigmoidActivationFunction);
            var it = (weightsIGate * VectorDColumn.Concatenation(htmo, data) + bI); it.ApplyFunction(sigmoidActivationFunction);
            var Cst = (weightsCGate * VectorDColumn.Concatenation(htmo, data) + bC);Cst.ApplyFunction(tanhActivationFunction);
            var Ct = ft & ctmo + it & Cst;
            var ot = ( weightsOutputGate * VectorDColumn.Concatenation(htmo,data) + bO ); ot.ApplyFunction(sigmoidActivationFunction);
            var ht = ot & (Ct.AppliedFunction(tanhActivationFunction));

            htmo = ht.GetColumn();
            ctmo = Ct.GetColumn();


            return ht.GetColumn();
        }


    }
}
