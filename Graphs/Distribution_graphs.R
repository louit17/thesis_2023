# Set your working directory to the GitHub/Speciale Folder
setwd("~/Desktop/GitHub/Speciale")
path_to_specific_folder = "~/Desktop/GitHub/Speciale/Graphs/Graphs_images"

# Insert slope and interception from python to get the most accurate values. 
slope = 0.07000588853139435
intercept = -4.620380213555453



#______________________________________________________________________________________________

library(ggplot2)
library(dplyr)
data <- read.csv('FinishedAnnotations_data.csv', header = T, sep = ',')
data <- data[data$Pathology.polyp.size != 0, ]

# 0-6 mm
sub_5 <- subset(data, Pathology.polyp.size <= 5 )
x_5 = (5-(intercept))/slope
x_5
#6-9 mm
sub_6to9 <- subset(data, Pathology.polyp.size <= 9  & Pathology.polyp.size >5 )
x_6to9 = (9-(intercept))/slope

#10-19
sub_10to19 <- subset(data, Pathology.polyp.size < 20 & Pathology.polyp.size >9 )
x_10to19 = (19-(intercept))/slope

#>20+
sub_20 <- subset(data, Pathology.polyp.size >19 )
x_20 = (19-(intercept))/slope


#Distribution plots
df_distribution <- bind_rows(list('1-5 mm'= sub_5,
                                  '6-9 mm' = sub_6to9 ,
                                  '10-19 mm' = sub_10to19,
                                  '20+ mm' = sub_20),
                             .id = "dataset")

# create the plot using ggplot2
df_distribution$dataset <- factor(df_distribution$dataset, levels = c("1-5 mm", "6-9 mm", "10-19 mm", "20+ mm"))

#band_colors <- c("red", "green", "blue", "orange")
distribution_plot <- ggplot(df_distribution, aes(x = Largest.diameter, fill = dataset)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("red", "orange", "green", "blue"), name = "Dataset")+
  labs(title = "",
       x = "Polyp pixel size",
       y = "Density")+ theme(plot.title = element_text(hjust = 0.5))
bw <- density(df_distribution$Largest.diameter, bw = df_distribution$layers[[1]]$stat_params$bw)$bw
print(paste0("Bandwidth = ", round(bw,2)))
dis_plot_size = 1
dis_plot_style = "dashed"

distribution_plot + geom_vline(xintercept = x_5, color = "black", linetype = dis_plot_style, size = dis_plot_size)+
  geom_vline(xintercept = x_6to9, color = "black", linetype = dis_plot_style, size = dis_plot_size)+geom_vline(xintercept = x_10to19, color = "black", linetype = dis_plot_style, size = dis_plot_size)+
  theme(legend.position="bottom")+theme(legend.title=element_blank())+
  #labs(caption = paste0("Bandwidth = ", round(bw,2)))+
  annotate("text", x =100 , y = 0.0075, label = "Interval 0-5 mm", size = 2)+
  annotate("text", x =170 , y = 0.0075, label = "Interval 6-9 mm",   size = 2)+
  annotate("text", x = x_10to19- ((x_10to19-x_6to9)/2) , y = 0.0075, label = "Interval 10-19 mm",   size = 2)+
  annotate("text", x =450 , y = 0.0075, label = "Interval 20+ mm",   size = 2)

setwd(path_to_specific_folder)
ggsave("distribution_plot.png", width = 20, height = 10, units = "cm")

library(gridtext)
library(gridExtra)
# Create a histogram with ggplot
bin5<-ggplot(df_distribution, aes(x = Largest.diameter, fill = dataset)) +
  geom_histogram(binwidth = 5, alpha = 0.5, position = "identity") +
  facet_grid(rows = vars(dataset), cols = vars()) +
  scale_fill_manual(values = c("red", "orange", "green", "blue"), name = "Dataset", labels=c('0-5 mm', '6-9 mm', '10-19 mm', '20+ mm')) +
  labs(title = "", x = "Polyp pixel size", y = "Count")+ theme(plot.title = element_text(hjust = 0.5))+
  annotation_custom(textbox_grob("Binwidth = 5"), 
                    xmin = 400, xmax = 600, ymin = 3, ymax = 5)+theme(legend.position="bottom")
ggsave("distribution_plot_bins5.png")

bin20<-ggplot(df_distribution, aes(x = Largest.diameter, fill = dataset)) +
  geom_histogram(binwidth = 20, alpha = 0.5, position = "identity") +
  facet_grid(rows = vars(dataset), cols = vars()) +
  scale_fill_manual(values = c("red", "green", "blue", "orange"), name = "Dataset", labels=c('0-5 mm', '6-9 mm', '10-19 mm', '20+ mm')) +
  labs(title = "", x = "Polyp pixel size", y = "Count")+ theme(plot.title = element_text(hjust = 0.5))+
  annotation_custom(textbox_grob("Binwidth = 20"), 
                    xmin = 400, xmax = 600, ymin = 8, ymax = 10)+theme(legend.position="bottom")
grid.arrange(bin5, bin20, ncol = 2)
ggsave("distribution_plot_bins20.png")


