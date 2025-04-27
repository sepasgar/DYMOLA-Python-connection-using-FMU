# DYMOLA-Python-connection-using-FMU
This script is a general guide to couple a DYMOLA simulation model with a Python script. This is mainly used when you want to make use of Deep/Machine learning in your simulation. The 'fmpy' library is used in this script, but there are also other libraries.
To generate FMU, you should export your simulation model in FMU format in DYMOLA or any other software that supports FMU. To set variables as Simulation input from your Python script, you need to use an 'INPUT' module in DYMOLA simulation as shown in Fig. 1 and to read data from the simulation, you need an 'OUTPUT' module as shown in Fig. 2.

<div align="center">

<table>
  <tr>
    <td align="center" style="vertical-align: top;">
      <img src="input.JPG" alt="Image 1" width="200" height="50"/><br/><br/>
      <sub>Fig. 1 - DYMOLA Input module, here named as 'window_state' for example</sub>
    </td>
    <td align="center" style="vertical-align: top;">
      <img src="output.JPG" alt="Image 2" width="200" height="80"/><br/>
      <sub>Fig. 2 - DYMOLA Output module here named as 'T_real' for example</sub>
    </td>
  </tr>
</table>

</div>

