#include "plugin.h"
#include "NeuralField_IP.h"
#include <cedar/processing/ElementDeclaration.h>
#include <cedar/processing/DataRole.h>

void pluginDeclaration(cedar::aux::PluginDeclarationListPtr plugin)
{
cedar::proc::ElementDeclarationPtr neuralField_IP_decl(
 new cedar::proc::ElementDeclarationTemplate<NeuralField_IP>("DFT"));

neuralField_IP_decl->setIconPath(":/field_ip.svg");

// define field plot
cedar::proc::ElementDeclaration::PlotDefinition field_plot_data("field plot", ":/cedar/dynamics/gui/field_plot.svg");
field_plot_data.appendData(cedar::proc::DataRole::BUFFER, "input sum");
field_plot_data.appendData(cedar::proc::DataRole::BUFFER, "activation_ip", true);
field_plot_data.appendData(cedar::proc::DataRole::BUFFER, "gain_ip", true);
field_plot_data.appendData(cedar::proc::DataRole::BUFFER, "bias_ip", true);
field_plot_data.appendData(cedar::proc::DataRole::OUTPUT, "activation_ip", true);
field_plot_data.appendData(cedar::proc::DataRole::OUTPUT, "sigmoided activation");
neuralField_IP_decl->definePlot(field_plot_data);

// define field plot again, but this time with image plots
cedar::proc::ElementDeclaration::PlotDefinition field_image_plot_data("field plot (image)", ":/cedar/dynamics/gui/field_image_plot.svg");
field_image_plot_data.mData.push_back(cedar::proc::PlotDataPtr(new cedar::proc::PlotData(cedar::proc::DataRole::BUFFER, "input sum", false, "cedar::aux::gui::ImagePlot")));
field_image_plot_data.mData.push_back(cedar::proc::PlotDataPtr(new cedar::proc::PlotData(cedar::proc::DataRole::BUFFER, "activation_ip", true, "cedar::aux::gui::ImagePlot")));
field_image_plot_data.mData.push_back(cedar::proc::PlotDataPtr(new cedar::proc::PlotData(cedar::proc::DataRole::OUTPUT, "activation_ip", true, "cedar::aux::gui::ImagePlot")));
field_image_plot_data.mData.push_back(cedar::proc::PlotDataPtr(new cedar::proc::PlotData(cedar::proc::DataRole::OUTPUT, "sigmoided activation", false, "cedar::aux::gui::ImagePlot")));
neuralField_IP_decl->definePlot(field_image_plot_data);

cedar::proc::ElementDeclaration::PlotDefinition kernel_plot_data("kernel", ":/cedar/dynamics/gui/kernel_plot.svg");
kernel_plot_data.appendData(cedar::proc::DataRole::BUFFER, "lateral kernel");
neuralField_IP_decl->definePlot(kernel_plot_data);

neuralField_IP_decl->setDefaultPlot("field plot");
    
plugin->add(neuralField_IP_decl);

}
