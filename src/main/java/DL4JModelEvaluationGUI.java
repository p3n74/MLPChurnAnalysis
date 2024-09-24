import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.*;
import java.awt.*;

public class DL4JModelEvaluationGUI {

    private XYSeries series;
    private ChartPanel chartPanel;

    public DL4JModelEvaluationGUI() {

        series = new XYSeries("Model Score");
    }

    public void createAndShowGUI() {

        JFrame frame = new JFrame("Training Progress");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);

        XYSeriesCollection dataset = new XYSeriesCollection(series);
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Training Progress",
                "Epoch",
                "Score",
                dataset
        );

        chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(800, 600));

        frame.getContentPane().add(chartPanel, BorderLayout.CENTER);

        frame.setVisible(true);
    }

    //update the chart with the score
    public void updateChart(int epoch, double score) {
        series.add(epoch, score);  // Add score to the series
        chartPanel.revalidate();   // Refresh the panel
        chartPanel.repaint();      // Force repaint to show the updated chart
    }

    //display evaluation results in a new GUI
    public void showEvaluationResults(String evalResults) {
        // Create a new JFrame for displaying evaluation results
        JFrame evalFrame = new JFrame("Evaluation Results");
        evalFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        evalFrame.setSize(600, 400);

        JTextArea evalTextArea = new JTextArea(evalResults);
        evalTextArea.setEditable(false);
        evalTextArea.setLineWrap(true);
        evalTextArea.setWrapStyleWord(true);

        JScrollPane scrollPane = new JScrollPane(evalTextArea);

        evalFrame.getContentPane().add(scrollPane, BorderLayout.CENTER);

        evalFrame.setVisible(true);
    }
}
