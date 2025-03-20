import React from 'react';
import { Box, Typography, Paper, Grid } from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Component for visualizing search result data
 * 
 * @param {Object} props - Component props
 * @param {Object} props.data - The search result data
 * @param {string} props.query - The search query
 */
const VisualizationComponent = ({ data, query }) => {
  // Format data for the source count chart
  const prepareSourceCountData = () => {
    if (!data || !data.sourceResults) return [];
    
    return Object.entries(data.sourceResults).map(([source, results]) => ({
      source: source === 'ads' ? 'NASA SciX' : 
              source === 'scholar' ? 'Google Scholar' : 
              source === 'semanticScholar' ? 'Semantic Scholar' : 
              source === 'webOfScience' ? 'Web of Science' : source,
      count: results.length
    }));
  };

  // Format data for the similarity metrics chart
  const prepareMetricsData = () => {
    if (!data || !data.metrics) return [];
    
    return Object.entries(data.metrics).map(([name, value]) => ({
      metric: name === 'jaccard' ? 'Jaccard Similarity' : 
              name === 'rankBiased' ? 'Rank-Biased Overlap' :
              name === 'cosine' ? 'Cosine Similarity' :
              name === 'euclidean' ? 'Euclidean Distance' : name,
      value: parseFloat(value.toFixed(3))
    }));
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Search Results Visualization
        </Typography>
        <Typography variant="body1" gutterBottom>
          Visualization for query: "{query}"
        </Typography>
        
        <Grid container spacing={3} sx={{ mt: 2 }}>
          {/* Results count by source */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom align="center">
              Results by Source
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={prepareSourceCountData()}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Legend />
                <Bar dataKey="count" fill="#8884d8" name="Number of Results" />
              </BarChart>
            </ResponsiveContainer>
          </Grid>
          
          {/* Similarity metrics */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom align="center">
              Similarity Metrics
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={prepareMetricsData()}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#82ca9d" name="Similarity Score" />
              </BarChart>
            </ResponsiveContainer>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default VisualizationComponent; 