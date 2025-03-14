import React, { useState, useEffect } from 'react';
import {
  Grid, Card, CardContent, CardHeader,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Typography,
  Box, Chip, Divider, IconButton, Tooltip
} from '@mui/material';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import LaunchIcon from '@mui/icons-material/Launch';
import InfoIcon from '@mui/icons-material/Info';

// Modified to automatically use selected sources from the initial query
const SideBySideComparison = ({ results, sourceResults }) => {
  // Extract available sources from sourceResults directly
  const availableSources = Object.keys(sourceResults || {}).filter(source => 
    sourceResults[source] && sourceResults[source].length > 0
  );
  
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
        return 'primary';
      case 'scholar':
        return 'error';
      case 'webOfScience':
        return 'success';
      case 'semanticScholar':
        return 'warning';
      default:
        return 'default';
    }
  };

  // Find matching results between selected sources
  const findMatches = (source1, source2) => {
    if (!source1 || !source2) return [];
    
    const results1 = sourceResults[source1] || [];
    const results2 = sourceResults[source2] || [];
    
    const matches = [];
    
    // First try matching by DOI
    results1.forEach(result1 => {
      if (result1.doi) {
        const matchByDoi = results2.find(r => r.doi === result1.doi);
        if (matchByDoi) {
          matches.push({
            source1: result1,
            source2: matchByDoi,
            matchType: 'doi'
          });
        }
      }
    });
    
    // Then try matching by normalized title for those without DOI matches
    const matchedDois = new Set(matches.map(m => m.source1.doi));
    results1.forEach(result1 => {
      if (!result1.doi || !matchedDois.has(result1.doi)) {
        const normalizedTitle1 = result1.title.toLowerCase().trim();
        const matchByTitle = results2.find(r => 
          r.title.toLowerCase().trim() === normalizedTitle1
        );
        
        if (matchByTitle) {
          matches.push({
            source1: result1,
            source2: matchByTitle,
            matchType: 'title'
          });
        }
      }
    });
    
    return matches;
  };

  // Extract results that only appear in one source
  const findUniqueResults = (source1, source2) => {
    if (!source1 || !source2) return { source1Only: [], source2Only: [] };
    
    const results1 = sourceResults[source1] || [];
    const results2 = sourceResults[source2] || [];
    
    const matches = findMatches(source1, source2);
    const matchedIds1 = new Set(matches.map(m => m.source1.doi || m.source1.title));
    const matchedIds2 = new Set(matches.map(m => m.source2.doi || m.source2.title));
    
    const source1Only = results1.filter(r => 
      !matchedIds1.has(r.doi || r.title)
    );
    
    const source2Only = results2.filter(r => 
      !matchedIds2.has(r.doi || r.title)
    );
    
    return { source1Only, source2Only };
  };

  // Calculate statistics
  const calculateStats = (source1, source2) => {
    if (!source1 || !source2) return null;
    
    const matches = findMatches(source1, source2);
    const { source1Only, source2Only } = findUniqueResults(source1, source2);
    
    return {
      source1Total: (sourceResults[source1] || []).length,
      source2Total: (sourceResults[source2] || []).length,
      matches: matches.length,
      source1Only: source1Only.length,
      source2Only: source2Only.length,
      matchByDoi: matches.filter(m => m.matchType === 'doi').length,
      matchByTitle: matches.filter(m => m.matchType === 'title').length
    };
  };

  // If there are not enough sources, show a message
  if (availableSources.length < 2) {
    return (
      <Card>
        <CardHeader title="Side-by-Side Comparison" />
        <CardContent>
          <Paper elevation={0} sx={{ p: 3, textAlign: 'center', bgcolor: 'background.default' }}>
            <Typography variant="body1" color="text.secondary">
              Please select at least two search sources in your initial query to see a side-by-side comparison.
            </Typography>
          </Paper>
        </CardContent>
      </Card>
    );
  }

  // Generate all possible pairs from available sources
  const generatePairs = (sources) => {
    const pairs = [];
    for (let i = 0; i < sources.length; i++) {
      for (let j = i + 1; j < sources.length; j++) {
        pairs.push([sources[i], sources[j]]);
      }
    }
    return pairs;
  };

  const sourcePairs = generatePairs(availableSources);

  // Render comparison section for a pair of sources
  const renderComparisonSection = (source1, source2) => {
    const stats = calculateStats(source1, source2);
    const matches = findMatches(source1, source2);
    const { source1Only, source2Only } = findUniqueResults(source1, source2);

    return (
      <Box sx={{ mb: 5 }}>
        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <Chip 
            label={formatSourceName(source1)} 
            color={getSourceColor(source1)} 
            sx={{ mr: 1 }} 
          />
          <CompareArrowsIcon sx={{ mx: 1 }} />
          <Chip 
            label={formatSourceName(source2)} 
            color={getSourceColor(source2)} 
          />
        </Typography>
        
        {/* Statistics */}
        <Paper elevation={1} sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                {formatSourceName(source1)} Total
              </Typography>
              <Typography variant="h6">
                {stats.source1Total}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                {formatSourceName(source2)} Total
              </Typography>
              <Typography variant="h6">
                {stats.source2Total}
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Matching Results
              </Typography>
              <Typography variant="h6">
                {stats.matches} 
                <Typography variant="caption" sx={{ ml: 1 }}>
                  ({stats.matchByDoi} by DOI, {stats.matchByTitle} by title)
                </Typography>
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" color="text.secondary">
                Unique Results
              </Typography>
              <Typography variant="h6">
                {stats.source1Only + stats.source2Only} 
                <Typography variant="caption" sx={{ ml: 1 }}>
                  ({stats.source1Only} in {formatSourceName(source1)}, 
                  {stats.source2Only} in {formatSourceName(source2)})
                </Typography>
              </Typography>
            </Grid>
          </Grid>
        </Paper>
        
        {/* Matching Results Table */}
        <Typography variant="subtitle1" gutterBottom sx={{ mt: 3, fontWeight: 'bold' }}>
          Matching Results ({matches.length})
        </Typography>
        <TableContainer component={Paper} sx={{ maxHeight: 300, mb: 3 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell>Title</TableCell>
                <TableCell>{formatSourceName(source1)} Rank</TableCell>
                <TableCell>{formatSourceName(source2)} Rank</TableCell>
                <TableCell>Match Type</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {matches.map((match, idx) => (
                <TableRow key={idx}>
                  <TableCell>
                    {match.source1.title}
                    {match.source1.url && (
                      <IconButton 
                        size="small" 
                        href={match.source1.url} 
                        target="_blank"
                        sx={{ ml: 1 }}
                      >
                        <LaunchIcon fontSize="small" />
                      </IconButton>
                    )}
                  </TableCell>
                  <TableCell>{match.source1.rank}</TableCell>
                  <TableCell>{match.source2.rank}</TableCell>
                  <TableCell>
                    <Chip 
                      label={match.matchType === 'doi' ? 'DOI' : 'Title'} 
                      size="small" 
                      color={match.matchType === 'doi' ? 'success' : 'primary'}
                    />
                  </TableCell>
                </TableRow>
              ))}
              {matches.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} align="center">
                    No matching results found between these sources.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
        
        {/* Unique Results */}
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
              Unique to {formatSourceName(source1)} ({source1Only.length})
            </Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Year</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {source1Only.map((result) => (
                    <TableRow key={result.title + result.rank}>
                      <TableCell>{result.rank}</TableCell>
                      <TableCell>
                        {result.title}
                        {result.url && (
                          <IconButton 
                            size="small" 
                            href={result.url} 
                            target="_blank"
                            sx={{ ml: 1 }}
                          >
                            <LaunchIcon fontSize="small" />
                          </IconButton>
                        )}
                      </TableCell>
                      <TableCell>{result.year}</TableCell>
                    </TableRow>
                  ))}
                  {source1Only.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} align="center">
                        No unique results found for this source.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
              Unique to {formatSourceName(source2)} ({source2Only.length})
            </Typography>
            <TableContainer component={Paper} sx={{ maxHeight: 300 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Title</TableCell>
                    <TableCell>Year</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {source2Only.map((result) => (
                    <TableRow key={result.title + result.rank}>
                      <TableCell>{result.rank}</TableCell>
                      <TableCell>
                        {result.title}
                        {result.url && (
                          <IconButton 
                            size="small" 
                            href={result.url} 
                            target="_blank"
                            sx={{ ml: 1 }}
                          >
                            <LaunchIcon fontSize="small" />
                          </IconButton>
                        )}
                      </TableCell>
                      <TableCell>{result.year}</TableCell>
                    </TableRow>
                  ))}
                  {source2Only.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} align="center">
                        No unique results found for this source.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>
        
        <Divider sx={{ my: 4 }} />
      </Box>
    );
  };

  return (
    <Card>
      <CardHeader 
        title="Side-by-Side Comparison" 
        subheader={`Comparing results from ${availableSources.length} sources`}
        action={
          <Tooltip title="This view shows matches and differences between search engines automatically based on your selected sources">
            <InfoIcon color="primary" />
          </Tooltip>
        }
      />
      <CardContent>
        {sourcePairs.map(([source1, source2], index) => (
          <React.Fragment key={`${source1}-${source2}`}>
            {renderComparisonSection(source1, source2)}
          </React.Fragment>
        ))}
      </CardContent>
    </Card>
  );
};

export default SideBySideComparison;