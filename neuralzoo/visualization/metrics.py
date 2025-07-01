from numpy import isclose, argwhere
from matplotlib.pyplot import plot, axvline, annotate, xlabel, ylabel, legend

def show_performance(training_result: list, metric: str, metric_label: str) -> None:

	train_perf = training_result.history[str(metric)]
	validation_perf = training_result.history['val_'+str(metric)]
	intersection_idx = argwhere(isclose(train_perf,
                                            	validation_perf, atol=1e-2)).flatten()[0]
	intersection_value = train_perf[intersection_idx]
    
	plot(train_perf, label=metric_label)
	plot(validation_perf, label = 'val_'+str(metric))
	axvline(x=intersection_idx, color='r', linestyle='--', label='Intersection')
    
	annotate(f'Optimal Value: {intersection_value:.4f}',
         	xy=(intersection_idx, intersection_value),
         	xycoords='data',
         	fontsize=10,
         	color='green')
            	 
	xlabel('Epoch')
	ylabel(metric_label)
	legend(loc='lower right')
	return