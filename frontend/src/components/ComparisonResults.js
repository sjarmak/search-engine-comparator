import React from 'react';
import {
  Card, CardContent, CardHeader,
  Grid, Typography, Box, Paper,
  List, ListItem, ListItemText, Divider,
  Chip, Avatar
} from '@mui/material';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

const ComparisonResults = ({ data }) => {
  if (!data) {
    return (
      <Card>
        <CardHeader title="Comparison Summary" />
        <CardContent>
          <Typography variant="body1">No comparison data available</Typography>
        </CardContent>
      </Card>
    );
  }

  // Format source name for display
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

  // Get source color
  const getSourceColor = (source) => {
    switch(source) {
      case 'ads':
        return '#E57373'; // Light red
      case 'scholar':
        return '#64B5F6'; // Light blue
      case 'webOfScience':
        return '#FFD54F'; // Light yellow
      case 'semanticScholar':
        return '#81C784'; // Light green
      default:
        return '#9E9E9E'; // Grey
    }
  };

  // Prepare data for the overview chart
  const prepareChartData = () => {
    const chartData = [];
    
    // Get sources
    const sources = Object.keys(data.sourceResults || {});
    
    // Count items per source
    sources.forEach(source => {
      const results = data.sourceResults[source] || [];
      chartData.push({
        name: formatSourceName(source),
        results: results.length,
        color: getSourceColor(source)
      });
    });
    
    return chartData;
  };

  // Prepare overlap statistics
  const prepareOverlapStats = () => {
    const overlapData = data.overlap || {};
    const stats = [];
    
    Object.entries(overlapData).forEach(([key, value]) => {
      const [source1, source2] = key.split('_vs_');
      
      stats.push({
        source1: formatSourceName(source1),
        source2: formatSourceName(source2),
        overlap: value.overlap,
        source1Only: value.source1_only,
        source2Only: value.source2_only,
        totalUnique: value.overlap + value.source1_only + value.source2_only
      });
    });
    
    return stats;
  };

  // Count unique results across all sources
  const countUniqueResults = () => {
    const allDois = new Set();
    const allTitles = new Set();
    
    Object.values(data.sourceResults || {}).forEach(results => {
      results.forEach(result => {
        if (result.doi) {
          allDois.add(result.doi);
        } else {
          // Fallback to title if DOI is not available
          allTitles.add(result.title.toLowerCase().trim());
        }
      });
    });
    
    return allDois.size + allTitles.size;
  };

  // Prepare top statistics
  const topStats = [
    {
      label: 'Total Sources',
      value: Object.keys(data.sourceResults || {}).length
    },
    {
      label: 'Total Results',
      value: data.allResults?.length || 0
    },
    {
      label: 'Unique Results',
      value: countUniqueResults()
    }
  ];

  return (
    <Card>
      <CardHeader title="Comparison Summary" />
      <CardContent>
        <Grid container spacing={3}>
          {/* Top Stats */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 3 }}>
              {topStats.map((stat) => (
                <Paper key={stat.label} elevation={2} sx={{ p: 2, minWidth: 160, textAlign: 'center' }}>
                  <Typography variant="h4" color="primary">
                    {stat.value}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {stat.label}
                  </Typography>
                </Paper>
              ))}
            </Box>
          </Grid>

          {/* Results per Source Chart */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Results per Source
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={prepareChartData()}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="results" name="Results">
                      {prepareChartData().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          </Grid>

          {/* Overlap Statistics */}
          <Grid item xs={12} md={6}>
            <Paper elevation={2} sx={{ p: 2, height: '100%', overflow: 'auto' }}>
              <Typography variant="h6" gutterBottom>
                Overlap Statistics
              </Typography>
              <List>
                {prepareOverlapStats().map((stat, index) => (
                  <React.Fragment key={index}>
                    {index > 0 && <Divider component="li" />}
                    <ListItem>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <Chip
                              size="small"
                              label={stat.source1}
                              sx={{ mr: 1 }}
                            />
                            <Typography variant="body2">vs</Typography>
                            <Chip
                              size="small"
                              label={stat.source2}
                              sx={{ ml: 1 }}
                            />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2" component="span" display="block">
                              Overlapping results: <strong>{stat.overlap}</strong>
                            </Typography>
                            <Typography variant="body2" component="span" display="block">
                              {stat.source1} only: <strong>{stat.source1Only}</strong>
                            </Typography>
                            <Typography variant="body2" component="span" display="block">
                              {stat.source2} only: <strong>{stat.source2Only}</strong>
                            </Typography>
                            <Typography variant="body2" component="span" display="block">
                              Total unique across both: <strong>{stat.totalUnique}</strong>
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  </React.Fragment>
                ))}
              </List>
            </Paper>
          </Grid>

          {/* Top Metrics Summary */}
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Key Insights
              </Typography>
              <Grid container spacing={2}>
                {Object.entries(data.metrics || {}).map(([key, metrics]) => {
                  const [source1, source2] = key.split('_vs_');
                  return (
                    <Grid item xs={12} sm={6} md={4} key={key}>
                      <Box sx={{ border: '1px solid #e0e0e0', borderRadius: 1, p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>
                          {formatSourceName(source1)} vs {formatSourceName(source2)}
                        </Typography>
                        {Object.entries(metrics).map(([metricName, value]) => (
                          <Typography key={metricName} variant="body2" sx={{ mb: 0.5 }}>
                            {metricName.charAt(0).toUpperCase() + metricName.slice(1).replace(/([A-Z])/g, ' $1')}: 
                            <strong> {typeof value === 'number' ? value.toFixed(4) : value}</strong>
                          </Typography>
                        ))}
                      </Box>
                    </Grid>
                  );
                })}
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

export default ComparisonResults;