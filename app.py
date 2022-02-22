from concurrent.futures import ThreadPoolExecutor
import pickle
import sys
from threading import Event

import numpy as np
import pydoocs
import PyQt5.QtCore as qtc
import PyQt5.QtGui as qtg
import PyQt5.QtWidgets as qtw
import pyqtgraph as pg
from scipy import constants
from tensorflow import keras

from nils.crisp_live_nils import get_charge, get_real_crisp_data
from nils.reconstruction_module import cleanup_formfactor, master_recon
from nils.simulate_spectrometer_signal import get_crisp_signal
import spectralvd


class CRISPThread(qtc.QThread):

    new_reading = qtc.pyqtSignal(str, np.ndarray, float)
    nbunch = 0
    
    def set_nbunch(self, n):
        self.nbunch = n
    
    def run(self):
        with ThreadPoolExecutor() as executor:
            while True:
                grating_future = executor.submit(CRISPThread.get_grating)
                charge_future = executor.submit(get_charge, shots=1)
                grating = grating_future.result()
                reading_future = executor.submit(get_real_crisp_data, 
                                                 shots=1, 
                                                 which_set="both", 
                                                 nbunch=self.nbunch)
                charge = charge_future.result()
                reading = reading_future.result()

                # TODO: Dummy data!
                # grating = "HIGH_FREQ"
                # reading = np.array([
                #     [
                #         6.84283010e+11, 6.86342276e+11, 6.89291238e+11, 6.93231268e+11,
                #         6.97867818e+11, 7.03183896e+11, 7.09148197e+11, 7.15734438e+11,
                #         7.23107481e+11, 7.31240520e+11, 7.40197234e+11, 7.50015830e+11,
                #         7.60740352e+11, 7.72411055e+11, 7.85016183e+11, 7.98657943e+11,
                #         8.13405379e+11, 8.29377459e+11, 8.46469777e+11, 8.65200287e+11,
                #         8.85430610e+11, 9.07299763e+11, 9.31292658e+11, 9.57343678e+11,
                #         9.85437362e+11, 1.01603519e+12, 1.04944212e+12, 1.08594355e+12,
                #         1.12511418e+12, 1.16825162e+12, 1.25452235e+12, 1.25941010e+12,
                #         1.26511563e+12, 1.27198544e+12, 1.28040955e+12, 1.29006363e+12,
                #         1.30097204e+12, 1.31295400e+12, 1.32651050e+12, 1.34139271e+12,
                #         1.35760346e+12, 1.37547358e+12, 1.39503008e+12, 1.41656334e+12,
                #         1.43974321e+12, 1.46457178e+12, 1.49167731e+12, 1.52101420e+12,
                #         1.55222847e+12, 1.58655331e+12, 1.62359096e+12, 1.66369362e+12,
                #         1.70718601e+12, 1.75508152e+12, 1.80652971e+12, 1.86252358e+12,
                #         1.92362000e+12, 1.99001009e+12, 2.06242973e+12, 2.14196419e+12,
                #         2.28119024e+12, 2.28903842e+12, 2.29950939e+12, 2.31261258e+12,
                #         2.32772918e+12, 2.34515972e+12, 2.36498904e+12, 2.38681143e+12,
                #         2.41113054e+12, 2.43827617e+12, 2.46831212e+12, 2.50106736e+12,
                #         2.53639935e+12, 2.57499338e+12, 2.61694409e+12, 2.66265777e+12,
                #         2.71189970e+12, 2.76495993e+12, 2.82181770e+12, 2.88430880e+12,
                #         2.95157763e+12, 3.02570939e+12, 3.10515027e+12, 3.19080363e+12,
                #         3.28411150e+12, 3.38505628e+12, 3.49557917e+12, 3.61569473e+12,
                #         3.74494818e+12, 3.87221178e+12, 3.87605079e+12, 3.89182771e+12,
                #         3.91112792e+12, 3.93302151e+12, 3.95880137e+12, 3.98757241e+12,
                #         4.02090481e+12, 4.05868059e+12, 4.09887219e+12, 4.14427297e+12,
                #         4.19590008e+12, 4.25159913e+12, 4.31184945e+12, 4.37798056e+12,
                #         4.45017789e+12, 4.52767991e+12, 4.61054903e+12, 4.70100308e+12,
                #         4.79815306e+12, 4.90363011e+12, 5.01847576e+12, 5.14289228e+12,
                #         5.27853835e+12, 5.42497233e+12, 5.58394521e+12, 5.75566213e+12,
                #         5.94270061e+12, 6.14675659e+12, 6.36808841e+12, 6.60398418e+12,
                #         6.84239080e+12, 6.86645607e+12, 6.89844358e+12, 6.93795435e+12,
                #         6.98381615e+12, 7.03665301e+12, 7.09581595e+12, 7.16182658e+12,
                #         7.23558808e+12, 7.31656935e+12, 7.40588391e+12, 7.50366620e+12,
                #         7.61017924e+12, 7.72725265e+12, 7.85313450e+12, 7.98952729e+12,
                #         8.13636493e+12, 8.29743559e+12, 8.46741872e+12, 8.65427554e+12,
                #         8.85635877e+12, 9.07462058e+12, 9.31463114e+12, 9.57583282e+12,
                #         9.85392517e+12, 1.01581001e+13, 1.04872716e+13, 1.08470040e+13,
                #         1.12261054e+13, 1.13980414e+13, 1.14448608e+13, 1.15007118e+13,
                #         1.15668861e+13, 1.16436805e+13, 1.16745582e+13, 1.17301474e+13,
                #         1.18286343e+13, 1.19375252e+13, 1.20584689e+13, 1.21926665e+13,
                #         1.23420542e+13, 1.25037214e+13, 1.26827481e+13, 1.28758007e+13,
                #         1.30836435e+13, 1.33098200e+13, 1.35576821e+13, 1.38245741e+13,
                #         1.41103234e+13, 1.44217346e+13, 1.47624195e+13, 1.51316322e+13,
                #         1.55258557e+13, 1.59552050e+13, 1.64196917e+13, 1.69285923e+13,
                #         1.74782780e+13, 1.80802622e+13, 1.87316513e+13, 1.94415752e+13,
                #         2.05304449e+13, 2.06041596e+13, 2.06980683e+13, 2.08140040e+13,
                #         2.09510109e+13, 2.11093505e+13, 2.12864874e+13, 2.14804199e+13,
                #         2.17008944e+13, 2.19489705e+13, 2.22216055e+13, 2.25161992e+13,
                #         2.28301018e+13, 2.31801788e+13, 2.35588868e+13, 2.39691136e+13,
                #         2.44111492e+13, 2.48897567e+13, 2.54005500e+13, 2.59618741e+13,
                #         2.65693466e+13, 2.72227653e+13, 2.79406515e+13, 2.87222610e+13,
                #         2.95520370e+13, 3.04799241e+13, 3.14692524e+13, 3.25463980e+13,
                #         3.36766529e+13, 3.41763020e+13, 3.43153008e+13, 3.44839431e+13,
                #         3.46803032e+13, 3.49137251e+13, 3.49825754e+13, 3.51760204e+13,
                #         3.54682908e+13, 3.57958567e+13, 3.61610300e+13, 3.65661507e+13,
                #         3.70094847e+13, 3.74942425e+13, 3.80275217e+13, 3.86055890e+13,
                #         3.92357545e+13, 3.99218111e+13, 4.06583840e+13, 4.14533934e+13,
                #         4.23052569e+13, 4.32362878e+13, 4.42368558e+13, 4.53735986e+13,
                #         4.65444168e+13, 4.78371060e+13, 4.92264177e+13, 5.07656062e+13,
                #         5.24232320e+13, 5.41042876e+13, 5.61221152e+13, 5.82673400e+13
                #     ],
                #     [
                #         1.18786128,  0.76746992,  0.82058976,  0.98040233,  0.99643733,
                #         0.94015949,  0.93740337,  0.93250338,  0.92030438,  0.93700321,
                #         0.98333393,  0.90148048,  0.95861757,  0.93402224,  0.94305785,
                #         1.17135232,  0.94580815,  0.93663307,  0.9364306 ,  0.97155708,
                #         0.95821595,  0.924865  ,  0.95561065,  0.92631002,  0.95444119,
                #         0.95747151,  0.93783878,  0.9264038 ,  0.90410405,  0.90219281,
                #         0.90457729,  0.80124367,  0.92353615,  0.83264104,  0.86462695,
                #         0.8930128 ,  0.95051366,  0.89856167,  0.87855058,  0.89753401,
                #         0.88843417,  0.8873796 ,  0.89604063,  0.86775456,  0.86515214,
                #         0.8563695 ,  0.87261302,  0.86898886,  0.86076269,  0.86344694,
                #         0.84793387,  0.84371862,  0.83203022,  0.8254263 ,  0.81458545,
                #         0.80655127,  0.77322346,  0.77246489,  0.76596685,  0.75980568,
                #         0.59243932,  0.69644209,  0.70492812,  0.74242301,  0.72103243,
                #         0.71658494,  0.70941435,  0.71450102,  0.71568424,  0.68291047,
                #         0.67490849,  0.68304372,  0.67262739,  0.65274671,  0.65470325,
                #         0.65933204,  0.63595744,  0.62559789,  0.61292721,  0.59290672,
                #         0.58180513,  0.57581109,  0.55237426,  0.55015866,  0.52138945,
                #         0.50067713,  0.48217297,  0.46253569,  0.44071071,  0.38705309,
                #         0.25596876,  0.44670354,  0.43579017,  0.41310491,  0.44941192,
                #         0.40094174,  0.39525097,  0.35470281,  0.36690945,  0.36320795,
                #         0.38049458,  0.37190423,  0.35336114,  0.35244628,  0.32845914,
                #         0.33298088,  0.32025266,  0.3178766 ,  0.30539195,  0.29477416,
                #         0.29340181,  0.28586541,  0.27391291,  0.26915315,  0.26045136,
                #         0.25924434,  0.25773021,  0.25712614,  0.25135646,  0.24680865,
                #         0.27099938,  0.28455402,  0.22652781,  0.2615593 ,  0.23300079,
                #         0.25717081,  0.24561495,  0.23554433,  0.23287802,  0.23463631,
                #         0.22692387,  0.21864089,  0.21649609,  0.22621241,  0.21103384,
                #         0.16462796,  0.20337286,  0.18909744,  0.19007792,  0.17448596,
                #         0.16534352,  0.16006147,  0.15268542,  0.14101591,  0.13683926,
                #         0.13496941,  0.13336211,  0.14165065,  0.13853947,  0.17336246,
                #         0.15349959,  0.09827239,  0.13993035,  0.15558818,  0.03761969,
                #         0.10303242,  0.14485108,  0.14227199,  0.12601624,  0.1478459 ,
                #         0.13414275,  0.13869831,  0.12447775,  0.11490487,  0.12805515,
                #         0.10304361,  0.12628933,  0.09858187,  0.08052734,  0.08131316,
                #         0.07538748,  0.06590053,  0.06524098,  0.05972144,  0.07271395,
                #         0.07321521,  0.07816205,  0.07230597,  0.0672394 ,  0.05372244,
                #         -0.08093983,  0.05786493,  0.03967606,  0.01845847,  0.06192408,
                #         0.02891736,  0.04560111,  0.04278758,  0.02180678,  0.07382971,
                #         0.01564263,  0.03992017,  0.04750721,  0.04276451,  0.04131807,
                #         0.03400447,  0.03762834,  0.03682845,  0.03082289,  0.02168744,
                #         0.01266831, -0.0168088 ,  0.02161535,  0.0200774 ,  0.02401339,
                #         0.0125904 ,  0.02551716,  0.01095764, -0.00865247, -0.11120871,
                #         -0.04791736, -0.02120429,  0.05096447, -0.0121157 , -0.05269058,
                #         0.0339602 , -0.03808444, -0.04122807,  0.03441391,  0.03284068,
                #         0.02069911,  0.04036188,  0.02603064,  0.01806063, -0.03415618,
                #         -0.03332522,  0.02354721, -0.02037137,  0.02400019, -0.03691792,
                #         -0.06250359,  0.01654372,  0.01471615, -0.01651185,  0.03578261,
                #         -0.01656503,  0.02150845, -0.01102986,  0.04491346, -0.04691778
                #     ], [
                #         0.35630182, 0.47523833, 0.61169844, 0.3646808 , 0.32503391,
                #         0.40774336, 0.1994366 , 0.28240715, 0.20833462, 0.20815378,
                #         0.15557378, 0.18068766, 0.14125349, 0.10410792, 0.08892758,
                #         0.66580467, 0.08232711, 0.07529163, 0.08162511, 0.07483875,
                #         0.07353107, 0.07706163, 0.07629928, 0.06307691, 0.06460088,
                #         0.05203473, 0.04873698, 0.05030488, 0.08742208, 0.08314344,
                #         0.37634332, 0.36928973, 0.16456054, 0.149908  , 0.12846789,
                #         0.09338067, 0.0837568 , 0.06752795, 0.06230983, 0.05594158,
                #         0.04201244, 0.0364838 , 0.03414032, 0.03019637, 0.02834899,
                #         0.02845096, 0.03280975, 0.0254627 , 0.02572024, 0.0244479 ,
                #         0.01802775, 0.01813355, 0.01547499, 0.01518672, 0.01668669,
                #         0.01838871, 0.02796345, 0.02256083, 0.01727761, 0.02061743,
                #         0.23879009, 0.16899433, 0.10261457, 0.06761464, 0.06991645,
                #         0.06702857, 0.04958954, 0.05356099, 0.04605814, 0.04656045,
                #         0.04911156, 0.03944464, 0.03757844, 0.03120333, 0.02571318,
                #         0.02531069, 0.02035075, 0.01855713, 0.01821075, 0.01735204,
                #         0.01671289, 0.01291185, 0.01033662, 0.01085429, 0.01108131,
                #         0.00877509, 0.01126018, 0.00935234, 0.0121493 , 0.03825784,
                #         0.33263956, 0.15915392, 0.12424294, 0.09950677, 0.07746555,
                #         0.07775379, 0.05755687, 0.06011699, 0.06432167, 0.05823304,
                #         0.05328085, 0.0502006 , 0.04668005, 0.03978955, 0.03724334,
                #         0.03144299, 0.02758832, 0.02276941, 0.01920431, 0.01730058,
                #         0.01551823, 0.01309447, 0.01011343, 0.00804605, 0.00802518,
                #         0.00712433, 0.00805078, 0.00749933, 0.00632021, 0.01939649,
                #         0.11176408, 0.0703241 , 0.06703529, 0.05239337, 0.04697282,
                #         0.03585647, 0.02933596, 0.02826133, 0.02575203, 0.02092107,
                #         0.02147195, 0.02182014, 0.01789051, 0.01267789, 0.01011808,
                #         0.57163142, 0.00810169, 0.00817419, 0.00752282, 0.00794374,
                #         0.00752479, 0.00629725, 0.00648687, 0.00689915, 0.006849  ,
                #         0.00606871, 0.00739143, 0.0079887 , 0.01180853, 0.08030292,
                #         0.07076998, 0.08881476, 0.05183124, 0.04415891, 0.96524899,
                #         0.06254737, 0.04645919, 0.04180744, 0.03552806, 0.02608889,
                #         0.02970758, 0.02739657, 0.03959431, 0.04020802, 0.03046446,
                #         0.05004315, 0.05860788, 0.06026753, 0.04432172, 0.02244878,
                #         0.01482656, 0.01176854, 0.00947373, 0.01021901, 0.00851208,
                #         0.00805491, 0.00627525, 0.00701023, 0.0072145 , 0.0139318 ,
                #         0.11240601, 0.09556924, 0.10156183, 0.16665498, 0.03006464,
                #         0.05299771, 0.0463053 , 0.09737417, 0.43969547, 0.10103905,
                #         0.09002989, 0.01581024, 0.015489  , 0.01432838, 0.016666  ,
                #         0.01851189, 0.03497734, 0.0139893 , 0.01544441, 0.02026212,
                #         0.0361933 , 0.02649155, 0.01859591, 0.01927406, 0.01738653,
                #         0.04445412, 0.01614629, 0.03769771, 0.05282188, 0.16465834,
                #         0.12298352, 0.13149512, 0.08202202, 0.15886407, 0.07853987,
                #         0.05698672, 0.09404538, 0.02702914, 0.02408803, 0.02037972,
                #         0.03999931, 0.02557153, 0.03425293, 0.05190818, 0.0390868 ,
                #         0.02845637, 0.02454675, 0.08026993, 0.02246246, 0.04134519,
                #         0.15576297, 0.06580023, 0.0971732 , 0.07408592, 0.02817878,
                #         0.02238229, 0.01676827, 0.0423157 , 0.08730571, 0.147986
                #     ], [
                #         0.51737672, 0.48028764, 0.56343877, 0.47552528, 0.45258935,
                #         0.49238999, 0.34385923, 0.40811077, 0.34822622, 0.35121875,
                #         0.31105251, 0.3209648 , 0.29264242, 0.24799076, 0.23030449,
                #         0.70231539, 0.22191566, 0.21118988, 0.21986935, 0.21444327,
                #         0.21109703, 0.21231138, 0.21474137, 0.1922332 , 0.19747349,
                #         0.17751072, 0.17002327, 0.17168021, 0.22358094, 0.21781042,
                #         0.46401278, 0.43259443, 0.31003077, 0.28096764, 0.26504931,
                #         0.22965311, 0.22439038, 0.19589833, 0.18607014, 0.17820005,
                #         0.15364434, 0.14309345, 0.13909536, 0.12873327, 0.12454608,
                #         0.12413494, 0.13456351, 0.11829716, 0.11832983, 0.11554566,
                #         0.09832565, 0.09836832, 0.09024013, 0.0890402 , 0.09271893,
                #         0.09685155, 0.11693995, 0.10498612, 0.09148751, 0.09953677,
                #         0.29911975, 0.27283091, 0.21389071, 0.17818087, 0.17855914,
                #         0.17429255, 0.14916262, 0.15557532, 0.14438727, 0.14180955,
                #         0.14478691, 0.13053689, 0.12643626, 0.1134979 , 0.10318467,
                #         0.10273515, 0.09047304, 0.08568762, 0.08402013, 0.0806647 ,
                #         0.07842049, 0.06857238, 0.06009259, 0.06145535, 0.06044935,
                #         0.05271324, 0.05859884, 0.05230555, 0.05819255, 0.0967744 ,
                #         0.2320574 , 0.2120476 , 0.1850502 , 0.16123952, 0.14838554,
                #         0.14041594, 0.11994995, 0.1161304 , 0.1221724 , 0.11565846,
                #         0.11323347, 0.10866383, 0.10213864, 0.09417724, 0.08795894,
                #         0.08137416, 0.0747522 , 0.06765815, 0.06090362, 0.05679238,
                #         0.05366208, 0.04865638, 0.04185723, 0.0370089 , 0.03635849,
                #         0.0341776 , 0.0362257 , 0.03492202, 0.03169755, 0.05502459,
                #         0.13840452, 0.1124992 , 0.09800032, 0.09309755, 0.08319879,
                #         0.07636768, 0.06750603, 0.06488549, 0.06158643, 0.05571916,
                #         0.05551251, 0.05492998, 0.04949387, 0.04258893, 0.03674854,
                #         0.24396349, 0.03228117, 0.03126656, 0.03007261, 0.0296079 ,
                #         0.02805148, 0.0252484 , 0.02502831, 0.02480543, 0.02434636,
                #         0.02276044, 0.02496867, 0.02675238, 0.0321662 , 0.09383356,
                #         0.08288828, 0.07429735, 0.06772777, 0.06591928, 0.15154523,
                #         0.063842  , 0.06523966, 0.06133403, 0.05321254, 0.04939096,
                #         0.0502033 , 0.04902287, 0.05583124, 0.05405558, 0.04967184,
                #         0.05710812, 0.06841894, 0.06129926, 0.04751109, 0.03397753,
                #         0.02658795, 0.02214729, 0.01977132, 0.01964647, 0.01978526,
                #         0.01931283, 0.01761281, 0.01790474, 0.01751581, 0.02175687,
                #         0.07585617, 0.05914006, 0.05048293, 0.04410847, 0.03431411,
                #         0.03113313, 0.03654416, 0.05133289, 0.07787298, 0.0686871 ,
                #         0.02984442, 0.01997931, 0.0215728 , 0.0196859 , 0.02086896,
                #         0.019953  , 0.02885133, 0.01805115, 0.01735151, 0.01667099,
                #         0.01702897, 0.01678174, 0.01594428, 0.01564427, 0.01624981,
                #         0.01881442, 0.01614238, 0.01616334, 0.01700169, 0.10761586,
                #         0.06104989, 0.04199347, 0.05141786, 0.03489009, 0.05115953,
                #         0.03498541, 0.04759458, 0.02654774, 0.02289722, 0.02057407,
                #         0.02288322, 0.0255493 , 0.02374686, 0.02435002, 0.02905793,
                #         0.02449014, 0.01911973, 0.03215893, 0.01846507, 0.03107035,
                #         0.07846927, 0.02623888, 0.0300736 , 0.02781511, 0.02525295,
                #         0.01531311, 0.01510301, 0.01718109, 0.04979945, 0.0662665
                #     ]
                # ])
                # charge = 250e-12

                self.new_reading.emit(grating, reading, charge)
    
    def get_grating():
        response = pydoocs.read("XFEL.SDIAG/THZ_SPECTROMETER.GRATINGMOVER/CRD.1934.TL/STATUS.STR")
        grating_raw = response["data"]
        return grating_raw[:-5].lower()


class ReconstructionThread(qtc.QThread):

    new_reconstruction = qtc.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._new_crisp_reading_event = Event()
        self._new_crisp_reading_event.clear()

        self._active_event = Event()
        self._active_event.set()

    def run(self):
        while True:
            self._active_event.wait()
            self._new_crisp_reading_event.wait()
            s, current = self.reconstruct()
            self._new_crisp_reading_event.clear()
            self.new_reconstruction.emit(s, current)

    def submit_reconstruction(self, grating, crisp_reading, charge):
        self._grating = grating
        self._crisp_reading = crisp_reading
        self._charge = charge

        self._new_crisp_reading_event.set()
    
    def set_active(self, active_state):
        if active_state:
            self._active_event.set()
        else:
            self._active_event.clear()
    
    def reconstruct(self):
        raise NotImplementedError


class NilsThread(ReconstructionThread):
    
    def reconstruct(self):
        frequency, formfactor, formfactor_noise, detlim = self._crisp_reading
        charge = self._charge

        t, current, _ = master_recon(frequency, formfactor, formfactor_noise, detlim, charge,
                                     method="KKstart", channels_to_remove=[], show_plots=False)

        s = t * constants.speed_of_light

        return s, current


class ANNThread(ReconstructionThread):

    def __init__(self, model_name):
        super().__init__()

        self.model = spectralvd.AdaptiveANNTHz.load("models/annthz")
    
    def reconstruct(self):
        frequency, formfactor, formfactor_noise, detlim = self._crisp_reading

        clean_frequency, clean_formfactor, _ = cleanup_formfactor(frequency, formfactor,
                                                                  formfactor_noise, detlim,
                                                                  channels_to_remove=[])

        prediction = self.model.predict([(clean_frequency, clean_formfactor)]*2)

        s = prediction[0][0]
        current = prediction[0][1]

        return s, current


class FormfactorPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        pen = pg.mkPen("c", width=2)
        self.plot_crisp = self.plot(range(999), np.ones(999), pen=pen, name="CRISP")
        # self.setXRange(int(684283010000), int(58267340000000))
        # self.setYRange(10e-3, 2)
        self.setLogMode(x=True, y=True)
        self.setLabel("bottom", text="Frequency", units="Hz")
        self.setLabel("left", text="|Frequency|")
        self.addLegend()
        self.showGrid(x=True, y=True)
    
    def update(self, grating, reading, charge):
        frequency, formfactor, _, _ = reading

        frequency_scaled = frequency.copy() # np.log10(frequency)
        formfactor_scaled = formfactor.copy()
        formfactor_scaled[formfactor_scaled <= 0] = 1e-3
        #formfactor_scaled = formfactor_scaled + 1e-12
        #formfactor_scaled = np.log10(formfactor_scaled + 1)

        self.plot_crisp.setData(frequency_scaled, formfactor_scaled)


class CurrentPlot(pg.PlotWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        limit = 0.00020095917745111108 # * 1e6
        s = np.linspace(-limit, limit, 100)
        
        ann_both_pen = pg.mkPen(qtg.QColor(255, 0, 0), width=3)
        
        ann_low_pen = pg.mkPen("g", width=3)

        nils_pen = pg.mkPen(qtg.QColor(0, 128, 255), width=3)
        self.ann_both_plot = self.plot(s, np.zeros(100), pen=ann_both_pen, name="ANN Both")
        self.ann_low_plot = self.plot(s, np.zeros(100), pen=ann_low_pen, name="ANN Low")
        self.nils_plot = self.plot(s, np.zeros(100), pen=nils_pen, name="Nils")

        self.setXRange(-limit, limit)
        self.setYRange(0, 10e3)
        self.setLabel("bottom", text="s", units="m")
        self.setLabel("left", text="Current", units="A")
        self.addLegend()
        self.showGrid(x=True, y=True)

        self._nils_hidden = False
        self._ann_both_hidden = False
        self._ann_low_hidden = False
    
    def update_ann_both(self, s, current):
        self.ann_both_s_scaled = s                # * 1e6
        self.ann_both_current_scaled = current    # * 1e-3

        if not self._ann_both_hidden:
            self.ann_both_plot.setData(self.ann_both_s_scaled, self.ann_both_current_scaled)
    
    def hide_ann_both(self, show):
        self._ann_both_hidden = not show
        if show:
            self.ann_both_plot.setData(self.ann_both_s_scaled, self.ann_both_current_scaled)
        else:
            # self.ann_both_plot.clear()
            self.ann_both_plot.setData([], [])

    def update_ann_low(self, s, current):
        self.ann_low_s_scaled = s                # * 1e6
        self.ann_low_current_scaled = current    # * 1e-3

        if not self._ann_low_hidden:
            self.ann_low_plot.setData(self.ann_low_s_scaled, self.ann_low_current_scaled)
    
    def hide_ann_low(self, show):
        self._ann_low_hidden = not show
        if show:
            self.ann_low_plot.setData(self.ann_low_s_scaled, self.ann_low_current_scaled)
        else:
            # self.ann_low_plot.clear()
            self.ann_low_plot.setData([], [])
    
    def update_nils(self, s, current):
        self.nils_s_scaled = s                # * 1e6
        self.nils_current_scaled = current    # * 1e-3

        if not self._nils_hidden:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
    
    def hide_nils(self, show):
        self._nils_hidden = not show
        if show:
            self.nils_plot.setData(self.nils_s_scaled, self.nils_current_scaled)
        else:
            # self.nils_plot.clear()
            self.nils_plot.setData([], [])


class App(qtw.QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectral Virtual Diagnostics at European XFEL")

        self.nils_checkbox = qtw.QCheckBox("Nils")
        self.nils_checkbox.setChecked(True)
        self.ann_both_checkbox = qtw.QCheckBox("ANN Both")
        self.ann_both_checkbox.setChecked(True)
        self.ann_low_checkbox = qtw.QCheckBox("ANN Low")
        self.ann_low_checkbox.setChecked(True)
        self.l1 = qtw.QLabel("N bunch: x2 ")
        self.sb_nbunch = qtw.QSpinBox()
        self.sb_nbunch.setMaximum(1024)
        

        self.formfactor_plot = FormfactorPlot()
        self.current_plot = CurrentPlot()

        self.crisp_thread = CRISPThread()
        self.nils_thread = NilsThread()
        self.ann_both_thread = ANNThread("both")
        self.ann_low_thread = ANNThread("low")

        self.crisp_thread.new_reading.connect(self.formfactor_plot.update)
        self.crisp_thread.new_reading.connect(self.nils_thread.submit_reconstruction)
        self.crisp_thread.new_reading.connect(self.ann_both_thread.submit_reconstruction)
        self.crisp_thread.new_reading.connect(self.ann_low_thread.submit_reconstruction)
        self.sb_nbunch.valueChanged.connect(self.crisp_thread.set_nbunch)
        
        self.nils_thread.new_reconstruction.connect(self.current_plot.update_nils)
        self.ann_both_thread.new_reconstruction.connect(self.current_plot.update_ann_both)
        self.ann_low_thread.new_reconstruction.connect(self.current_plot.update_ann_low)

        self.nils_checkbox.stateChanged.connect(self.nils_thread.set_active)
        self.nils_checkbox.stateChanged.connect(self.current_plot.hide_nils)
        self.ann_both_checkbox.stateChanged.connect(self.ann_both_thread.set_active)
        self.ann_both_checkbox.stateChanged.connect(self.current_plot.hide_ann_both)
        self.ann_low_checkbox.stateChanged.connect(self.ann_low_thread.set_active)
        self.ann_low_checkbox.stateChanged.connect(self.current_plot.hide_ann_low)

        grid = qtw.QGridLayout()
        grid.addWidget(self.formfactor_plot, 0, 0, 1, 3)
        grid.addWidget(self.current_plot, 0, 3, 1, 3)
        grid.addWidget(self.ann_both_checkbox, 1, 3, 1, 1)
        grid.addWidget(self.ann_low_checkbox, 1, 4, 1, 1)
        grid.addWidget(self.nils_checkbox, 1, 5, 1, 1)
        grid.addWidget(self.l1, 1, 0, 1, 1)
        grid.addWidget(self.sb_nbunch, 1, 1, 1, 1)

        self.setLayout(grid)

        self.crisp_thread.start()
        self.nils_thread.start()
        self.ann_both_thread.start()
        self.ann_low_thread.start()

    def handle_application_exit(self):
        pass
    

if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)

    # Force the style to be the same on all OSs
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors
    palette = qtg.QPalette()
    palette.setColor(qtg.QPalette.Window, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.WindowText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Base, qtg.QColor(25, 25, 25))
    palette.setColor(qtg.QPalette.AlternateBase, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ToolTipBase, qtc.Qt.white)
    palette.setColor(qtg.QPalette.ToolTipText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Text, qtc.Qt.white)
    palette.setColor(qtg.QPalette.Button, qtg.QColor(53, 53, 53))
    palette.setColor(qtg.QPalette.ButtonText, qtc.Qt.white)
    palette.setColor(qtg.QPalette.BrightText, qtc.Qt.red)
    palette.setColor(qtg.QPalette.Link, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.Highlight, qtg.QColor(42, 130, 218))
    palette.setColor(qtg.QPalette.HighlightedText, qtc.Qt.black)
    app.setPalette(palette)

    window = App()
    window.show()

    app.aboutToQuit.connect(window.handle_application_exit)

    sys.exit(app.exec_())
