using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using MathOps;
using System.IO;


namespace TestApp
{
    static class Program
    {
        
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            
            Random ra = new Random();
            MatrixD a = new MatrixD(100, 50);
            MatrixD b = new MatrixD(50, 400);

            for(int i = 0; i < a.Rows; ++ i)
            {
                for(int j = 0; j < a.Columns; ++ j)
                {
                    a[i, j] = ra.NextDouble() - ra.NextDouble();
                }
            }

            for (int i = 0; i < b.Rows; ++i)
            {
                for (int j = 0; j < b.Columns; ++j)
                {
                    b[i, j] = ra.NextDouble() - ra.NextDouble();
                }
            }


            MatrixD r = a * b;

            using (Stream s = File.Create("newfile.bin"))
            {
                MatrixD.WriteToStream(r, s);
            }



            /*Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Form1());*/
        }
    }
}
