import React, { useState } from 'react';
import { 
  Container, Box, Typography, TextField, Button, 
  Checkbox, FormControlLabel, FormGroup, Grid, 
  CircularProgress, Paper, Tabs, Tab, Divider, Alert 
} from '@mui/material';
import ComparisonResults from './components/ComparisonResults';
import VennDiagram from './components/VennDiagram';
import MetricsTable from './components/MetricsTable';
import ResultsTable from './components/ResultsTable';
import SciXModifier from './components/SciXModifier';
import SideBySideComparison from './components/SideBySideComparison';

const API_URL = process.env.REACT_APP_API_URL || 'https://search-engine-comparator.onrender.com';

function App() {
  // State for search query and options
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [tabValue, setTabValue] = useState(0);
  
  // State for source selection
  const [sources, setSources] = useState({
    ads: true,
    scholar: true,
    webOfScience: false,
    semanticScholar: false
  });
  
  // State for similarity metrics selection
  const [metrics, setMetrics] = useState({
    jaccard: true,
    rankBased: true,
    cosine: false,
    euclidean: false
  });
  
  // State for metadata fields to compare
  const [fields, setFields] = useState({
    title: true,
    abstract: false,
    authors: false,
    doi: true,
    year: false
  });

  // Handle source selection changes
  const handleSourceChange = (event) => {
    setSources({
      ...sources,
      [event.target.name]: event.target.checked
    });
  };

  // Handle metrics selection changes
  const handleMetricsChange = (event) => {
    setMetrics({
      ...metrics,
      [event.target.name]: event.target.checked
    });
  };

  // Handle fields selection changes
  const handleFieldsChange = (event) => {
    setFields({
      ...fields,
      [event.target.name]: event.target.checked
    });
  };

  // Handle tab changes
  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  // Submit the search query
  const handleSearch = async () => {
    if (!query.trim()) {
      setError("Please enter a search query");
      return;
    }
    
    if (!Object.values(sources).some(val => val)) {
      setError("Please select at least one search source");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const selectedSources = Object.keys(sources).filter(key => sources[key]);
      const selectedMetrics = Object.keys(metrics).filter(key => metrics[key]);
      const selectedFields = Object.keys(fields).filter(key => fields[key]);
      
      const requestBody = {
        query,
        sources: selectedSources,
        metrics: selectedMetrics,
        fields: selectedFields
      };
      
      const response = await fetch(`${API_URL}/api/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(`Failed to fetch results: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box my={4}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          Academic Search Engine Comparisons
        </Typography>
        
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Box component="form" noValidate autoComplete="off">
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Search Query"
                  variant="outlined"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Enter your academic search query"
                />
              </Grid>
              
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle1">Search Sources</Typography>
                <FormGroup>
                  <FormControlLabel
                    control={<Checkbox checked={sources.ads} onChange={handleSourceChange} name="ads" />}
                    label="ADS/SciX"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={sources.scholar} onChange={handleSourceChange} name="scholar" />}
                    label="Google Scholar"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={sources.webOfScience} onChange={handleSourceChange} name="webOfScience" />}
                    label="Web of Science"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={sources.semanticScholar} onChange={handleSourceChange} name="semanticScholar" />}
                    label="Semantic Scholar"
                  />
                </FormGroup>
              </Grid>
              
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle1">Similarity Metrics</Typography>
                <FormGroup>
                  <FormControlLabel
                    control={<Checkbox checked={metrics.jaccard} onChange={handleMetricsChange} name="jaccard" />}
                    label="Jaccard Similarity"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={metrics.rankBased} onChange={handleMetricsChange} name="rankBased" />}
                    label="Rank-Based Overlap"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={metrics.cosine} onChange={handleMetricsChange} name="cosine" />}
                    label="Cosine Similarity"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={metrics.euclidean} onChange={handleMetricsChange} name="euclidean" />}
                    label="Euclidean Distance"
                  />
                </FormGroup>
              </Grid>
              
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle1">Metadata Fields</Typography>
                <FormGroup>
                  <FormControlLabel
                    control={<Checkbox checked={fields.title} onChange={handleFieldsChange} name="title" />}
                    label="Title"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={fields.abstract} onChange={handleFieldsChange} name="abstract" />}
                    label="Abstract"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={fields.authors} onChange={handleFieldsChange} name="authors" />}
                    label="Authors"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={fields.doi} onChange={handleFieldsChange} name="doi" />}
                    label="DOI"
                  />
                  <FormControlLabel
                    control={<Checkbox checked={fields.year} onChange={handleFieldsChange} name="year" />}
                    label="Publication Year"
                  />
                </FormGroup>
              </Grid>
              
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleSearch}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? <CircularProgress size={24} /> : "Compare Search Results"}
                </Button>
              </Grid>
            </Grid>
          </Box>
        </Paper>

        {results && (
          <Box mt={4}>
            {Object.keys(results.sourceResults).length === 0 && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                No results could be retrieved. Please try a different query or check your API settings.
              </Alert>
            )}

            {Object.keys(results.sourceResults).some(source => results.sourceResults[source].length === 0) && (
              <Alert severity="info" sx={{ mb: 2 }}>
                Some selected sources returned no results or encountered errors.
              </Alert>
            )}

            <Tabs value={tabValue} onChange={handleTabChange} centered sx={{ mb: 3 }}>
              <Tab label="Overview" />
              <Tab label="Detailed Results" />
              <Tab label="Side-by-Side Comparison" />
              <Tab label="ADS/SciX Modifier" />
            </Tabs>
            
            <Box role="tabpanel" hidden={tabValue !== 0}>
              {tabValue === 0 && (
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <ComparisonResults data={results} />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <VennDiagram data={results.overlap} />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <MetricsTable metrics={results.metrics} />
                  </Grid>
                </Grid>
              )}
            </Box>
            
            <Box role="tabpanel" hidden={tabValue !== 1}>
              {tabValue === 1 && (
                <ResultsTable results={results.allResults} />
              )}
            </Box>
            
            <Box role="tabpanel" hidden={tabValue !== 2}>
              {tabValue === 2 && (
                <SideBySideComparison 
                  results={results.allResults} 
                  sourceResults={results.sourceResults}
                />
              )}
            </Box>
            
            <Box role="tabpanel" hidden={tabValue !== 3}>
              {tabValue === 3 && (
                <SciXModifier 
                  originalResults={results.sourceResults.ads || []} 
                  query={query}
                />
              )}
            </Box>
          </Box>
        )}

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </Box>
    </Container>
  );
}

export default App;
