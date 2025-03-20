import React from 'react';
import { 
  Card, CardContent, CardHeader, 
  Table, TableBody, TableCell, 
  TableContainer, TableHead, TableRow,
  Paper, Tooltip, Typography
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

const MetricsTable = ({ metrics }) => {
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <Card>
        <CardHeader title="Similarity Metrics" />
        <CardContent>
          <Typography variant="body1">No metrics data available</Typography>
        </CardContent>
      </Card>
    );
  }

  // Helper function to format the comparison key for display
  const formatSourceName = (source) => {
    switch(source) {
      case 'ads':
        return 'ADS/SciX';
      case 'scholar':
        return 'Google Scholar';
      case 'webOfScience':
        return 'Web of Science';
      case 'semanticScholar':
        return 'Semantic Scholar';
      default:
        return source.charAt(0).toUpperCase() + source.slice(1);
    }
  };

  const formatComparisonKey = (key) => {
    const [source1, source2] = key.split('_vs_');
    return `${formatSourceName(source1)} vs. ${formatSourceName(source2)}`;
  };

  // Get all metric types from the data
  const metricTypes = [];
  Object.values(metrics).forEach(metricObj => {
    Object.keys(metricObj).forEach(metricType => {
      if (!metricTypes.includes(metricType)) {
        metricTypes.push(metricType);
      }
    });
  });

  // Helper function to format metric names
  const formatMetricName = (metric) => {
    switch(metric) {
      case 'jaccard':
        return 'Jaccard Similarity';
      case 'rankBiased':
        return 'Rank-Biased Overlap';
      case 'cosine':
        return 'Cosine Similarity';
      case 'euclidean':
        return 'Euclidean Distance';
      default:
        return metric.charAt(0).toUpperCase() + metric.slice(1);
    }
  };

  // Helper function to get tooltip description for each metric
  const getMetricDescription = (metric) => {
    switch(metric) {
      case 'jaccard':
        return 'Measures the similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets.';
      case 'rankBiased':
        return 'Rank-Biased Overlap (RBO) by Webber et al. (2010). Measures the similarity between two ranked lists, weighting items towards the top of the lists more heavily using a persistence parameter p=0.9.';
      case 'cosine':
        return 'Measures the cosine of the angle between two vectors, representing how similar the two vectors are irrespective of their size.';
      case 'euclidean':
        return 'Measures the straight-line distance between two points in Euclidean space.';
      default:
        return 'No description available for this metric.';
    }
  };

  // Helper function to format metric values
  const formatMetricValue = (value) => {
    return typeof value === 'number' ? value.toFixed(4) : value;
  };

  return (
    <Card>
      <CardHeader 
        title="Similarity Metrics" 
        action={
          <Tooltip title="Higher values indicate greater similarity between search results (except for Euclidean distance, where lower values indicate greater similarity).">
            <InfoIcon color="primary" />
          </Tooltip>
        }
      />
      <CardContent>
        <TableContainer component={Paper} sx={{ maxHeight: 380 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Comparison</TableCell>
                {metricTypes.map(metric => (
                  <TableCell key={metric} align="right">
                    <Tooltip title={getMetricDescription(metric)}>
                      <Typography variant="body2" display="inline" sx={{ cursor: 'help', textDecoration: 'underline', textDecorationStyle: 'dotted' }}>
                        {formatMetricName(metric)}
                      </Typography>
                    </Tooltip>
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {Object.entries(metrics).map(([comparisonKey, metricValues]) => (
                <TableRow key={comparisonKey}>
                  <TableCell component="th" scope="row">
                    {formatComparisonKey(comparisonKey)}
                  </TableCell>
                  {metricTypes.map(metric => (
                    <TableCell key={`${comparisonKey}-${metric}`} align="right">
                      {metric in metricValues ? formatMetricValue(metricValues[metric]) : '—'}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </CardContent>
    </Card>
  );
};

export default MetricsTable;