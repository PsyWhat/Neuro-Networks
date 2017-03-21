using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathOps;

namespace LSTM
{

    public class TeachingOutput
    {
        VectorDColumn _res;
        bool _resMatter;

        public VectorDColumn Result
        {
            get
            {
                if(_resMatter)
                {
                    return _res;
                }else
                {
                    return null;
                }
            }
        }

        public bool ResMatter
        {
            get
            {
                return _resMatter;
            }
        }

        public TeachingOutput()
        {
            _res = null;
            _resMatter = false;
        }

        public TeachingOutput(VectorDColumn Result)
        {
            _res = new VectorDColumn(Result);
            _resMatter = true;
        }


    }

    public class TeachingSequence:List<Tuple<VectorDColumn,TeachingOutput>>
    {
        public TeachingSequence()
        {

        }
    }

    public class LSTMLayerState
    {
        public VectorDColumn xt;
        public VectorDColumn ht;
        public VectorDColumn htmo;
        public VectorDColumn ct;
        public VectorDColumn ctmo;

        public VectorDColumn at;
        public VectorDColumn it;
        public VectorDColumn ft;
        public VectorDColumn ot;


        public VectorDColumn ats;
        public VectorDColumn its;
        public VectorDColumn fts;


        public MatrixD Wc;
        public MatrixD Wi;
        public MatrixD Wf;
        public MatrixD Wo;

        public MatrixD Uc;
        public MatrixD Ui;
        public MatrixD Uf;
        public MatrixD Uo;

    }

    public class LSTMBackPropRes
    {
        public MatrixD dWc;
        public MatrixD dWi;
        public MatrixD dWf;
        public MatrixD dWo;

        public MatrixD dUc;
        public MatrixD dUi;
        public MatrixD dUf;
        public MatrixD dUo;

        public VectorDColumn dCtmo;
        public VectorDColumn dHtmo;

        public LSTMBackPropRes(VectorDColumn deltaHtmo, VectorDColumn deltaCtmo, 
            MatrixD deltaWc, MatrixD deltaWi, MatrixD deltaWf, MatrixD deltaWo, 
            MatrixD deltaUc, MatrixD deltaUi, MatrixD deltaUf, MatrixD deltaUo
            )
        {
            this.dWc = deltaWc;
            this.dWi = deltaWi;
            this.dWf = deltaWf;
            this.dWo = deltaWo;

            this.dUc = deltaUc;
            this.dUi = deltaUi;
            this.dUf = deltaUf;
            this.dUo = deltaUo;

            this.dCtmo = deltaCtmo;
            this.dHtmo = deltaHtmo;
        }

        public static LSTMBackPropRes operator+(LSTMBackPropRes a, LSTMBackPropRes b)
        {
            LSTMBackPropRes res = new LSTMBackPropRes(a.dHtmo + b.dHtmo,a.dCtmo + b.dCtmo, 
                a.dWc + b.dWc, a.dWi+b.dWi,a.dWf + b.dWf,a.dWo + b.dWo,
                a.dUc + b.dUc, a.dUi + b.dUi, a.dUf+b.dUf,a.dUo + b.dUo
                );
            return res;
        }

        public static LSTMBackPropRes operator *(LSTMBackPropRes a, double d)
        {
            LSTMBackPropRes res = new LSTMBackPropRes(a.dHtmo, a.dCtmo,
                a.dWc * d, a.dWi * d, a.dWf * d, a.dWo * d,
                a.dUc * d, a.dUi * d, a.dUf * d, a.dUo * d
                );
            return res;
        }
        public static LSTMBackPropRes operator *(double d, LSTMBackPropRes a)
        {
            LSTMBackPropRes res = new LSTMBackPropRes(a.dHtmo, a.dCtmo,
                a.dWc * d, a.dWi * d, a.dWf * d, a.dWo * d,
                a.dUc * d, a.dUi * d, a.dUf * d, a.dUo * d
                );
            return res;
        }

    }

    public class LSTMLayer
    {
        MatrixD Wc;
        MatrixD Wi;
        MatrixD Wf;
        MatrixD Wo;

        MatrixD Uc;
        MatrixD Ui;
        MatrixD Uf;
        MatrixD Uo;



        VectorDColumn ctmo;

        VectorDColumn htmo;

        int numInputs;
        int numOutputs;

        LSTMLayer(int InputsCount, int OutputsCount)
        {
            numInputs = InputsCount;
            numOutputs = OutputsCount;

            Wc = new MatrixD(OutputsCount, InputsCount);
            Uc = new MatrixD(OutputsCount, OutputsCount);

            Wi = new MatrixD(OutputsCount, InputsCount);
            Ui = new MatrixD(OutputsCount, OutputsCount);

            Wf = new MatrixD(OutputsCount, InputsCount);
            Uf = new MatrixD(OutputsCount, OutputsCount);

            Wo = new MatrixD(OutputsCount, InputsCount);
            Uo = new MatrixD(OutputsCount, OutputsCount);

            ctmo = new VectorDColumn(OutputsCount);
            htmo = new VectorDColumn(OutputsCount);
        }

        public void RandomizeWeights(double from = -1.0, double to = 1.0)
        {
            Random r = new Random();
            Func< int, int, double> initializer = (i, j) => {
                return r.NextDouble()*(to - from) + from;
            };
            Wc.InitElements(initializer);
            Uc.InitElements(initializer);

            Wi.InitElements(initializer);
            Ui.InitElements(initializer);

            Wf.InitElements(initializer);
            Uf.InitElements(initializer);

            Wo.InitElements(initializer);
            Uo.InitElements(initializer);
        }

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
        
        public void SetMemory(VectorDColumn C, VectorDColumn H)
        {
            ctmo = new VectorDColumn(C);
            htmo = new VectorDColumn(H);
        }

        public void FlushMemory()
        {
            ctmo = new VectorDColumn(numOutputs);
            htmo = new VectorDColumn(numOutputs);
        }

        public VectorDColumn Process(VectorDColumn data)
        {

            var at = (Wc * data + Uc * htmo); at.ApplyFunction(tanhActivationFunction);
            var it = (Wi * data + Ui * htmo); it.ApplyFunction(sigmoidActivationFunction);
            var ft = (Wf * data + Uf * htmo); ft.ApplyFunction(sigmoidActivationFunction);
            var ot = (Wo * data + Uo * htmo); ot.ApplyFunction(sigmoidActivationFunction);

            var ct = it & at + ft & ctmo;
            var ht = ot & ct.AppliedFunction(tanhActivationFunction);

            this.ctmo = ct;
            this.htmo = ht;

            return new VectorDColumn(ht);
        }

        public LSTMLayerState ForwardPass(VectorDColumn data)
        {
            LSTMLayerState state = new LSTMLayerState();

            var ats = (Wc * data + Uc * htmo);
            var its = (Wi * data + Ui * htmo);
            var fts = (Wf * data + Uf * htmo);
            var ots = (Wo * data + Uo * htmo);

            state.ats = ats;
            state.its = its;
            state.fts = fts;

            var at = ats.AppliedFunction(tanhActivationFunction);
            var it = its.AppliedFunction(sigmoidActivationFunction);
            var ft = fts.AppliedFunction(sigmoidActivationFunction);
            var ot = ots.AppliedFunction(sigmoidActivationFunction);

            state.at = at;
            state.it = it;
            state.ft = ft;
            state.ot = ot;

            var ct = it & at + ft & ctmo;
            var ht = ot & ct.AppliedFunction(tanhActivationFunction);

            state.xt = new VectorDColumn(data);
            state.ctmo = new VectorDColumn(ct);
            state.ht = new VectorDColumn(ht);
            state.htmo = new VectorDColumn(this.htmo);

            this.htmo = ht;
            this.ctmo = ct;


            state.Wc = new MatrixD(Wc);
            state.Uc = new MatrixD(Uc);

            state.Wf = new MatrixD(Wf);
            state.Uf = new MatrixD(Uf);

            state.Wi = new MatrixD(Wi);
            state.Ui = new MatrixD(Ui);

            state.Wo = new MatrixD(Wo);
            state.Uo = new MatrixD(Uo);

            return state;
            
        }

        public static LSTMBackPropRes BackBpop( LSTMLayerState state , double LerningSpeed, VectorDColumn Herror, VectorDColumn Cerror)
        {
            var deltaOt = Herror & state.ct.AppliedFunction(tanhActivationFunction);
            var deltaCt = Herror & deltaOt & (state.ct.AppliedFunction(tanhDiffActivationFunction)) + Cerror;

            var deltaIt = deltaCt & state.at;
            var deltaFt = deltaCt & state.ctmo;
            var deltaAt = deltaCt & state.it;
            var deltaCtmo = deltaCt & state.ft;

            var deltaAts = deltaAt & state.ats.AppliedFunction(tanhDiffActivationFunction);
            var deltaIts = deltaIt & state.it.AppliedFunction(sigmoidDiffActivationFunction);
            var deltaFts = deltaFt & state.ft.AppliedFunction(sigmoidDiffActivationFunction);
            var deltaOts = deltaOt & state.ot.AppliedFunction(sigmoidDiffActivationFunction);

            var deltaXt = state.Wc * deltaAts + state.Wf * deltaFts + state.Wi * deltaIts + state.Wo * deltaOts;
            var deltaHtmo = state.Uc * deltaAts + state.Uf * deltaFts + state.Ui * deltaIts + state.Wo * deltaOts;

            var deltaWc = deltaAts * state.xt;
            var deltaWi = deltaIts * state.xt;
            var deltaWf = deltaFts * state.xt;
            var deltaWo = deltaOts * state.xt;


            var deltaUc = deltaAts * state.htmo;
            var deltaUi = deltaIts * state.htmo;
            var deltaUf = deltaFts * state.htmo;
            var deltaUo = deltaOts * state.htmo;

            LSTMBackPropRes res = new LSTMBackPropRes(deltaHtmo, deltaCtmo, deltaWc, deltaWi, deltaWf, deltaWo, deltaUc, deltaUi, deltaUf, deltaUo);
            return res;
        }

        public void ApplyBackPropRes(LSTMBackPropRes res)
        {
            this.Wc += res.dWc;
            this.Uc += res.dUc;

            this.Wi += res.dWi;
            this.Ui += res.dUi;

            this.Wf += res.dWf;
            this.Uf += res.dUf;

            this.Wo += res.dWo;
            this.Uo += res.dUo;
        }




    }
}
