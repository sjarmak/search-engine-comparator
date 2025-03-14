import React, { useMemo, useState } from 'react';
import {
  Box, Card, CardContent, CardHeader,
  Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Typography,
  TextField, Chip, IconButton, Link,
  MenuItem, Select, FormControl, InputLabel,
  Grid, Divider, Tabs, Tab
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import SortIcon from '@mui/icons-material/Sort';
import LaunchIcon from '@mui/icons-material/Launch';

const ResultsTable = ({ results }) => {
  // Initialize state at the top level
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('rank');
  const [sortOrder, setSortOrder] = useState('asc');
  const [currentTab, setCurrentTab] = useState(0);

  // Define columns unconditionally
  const columns = useMemo(() => [
    { id: 'rank', label: 'Rank', minWidth: 50 },
    { id: 'title', label: 'Title', minWidth: 200 },
    { id: 'authors', label: 'Authors', minWidth: 170 },
    { id: 'year', label: 'Year', minWidth: 50 }
  ], []);

  // Process results unconditionally
  const processedResults = useMemo(() => {
    if (!results || results.length === 0) return [];
    return [...results].sort((a, b) => a.rank - b.rank);
  }, [results]);

  // Group results by source - MOVED UP before conditional return
  const resultsBySource = useMemo(() => {
    if (!results || results.length === 0) return {};
    
    const grouped = {};
    results.forEach(result => {
      if (!grouped[result.source]) {
        grouped[result.source] = [];
      }
      grouped[result.source].push(result);
    });
    
    return grouped;
  }, [results]);

  // Get available sources - MOVED UP before conditional return
  const sources = useMemo(() => {
    if (!results || results.length === 0) return [];
    return Object.keys(resultsBySource);
  }, [resultsBySource]);

  // Apply filtering and sorting to specific source results
  const getFilteredAndSortedResults = (sourceResults) => {
    if (!sourceResults) return [];
    
    let filtered = [...sourceResults];
    
    // Apply search term filter
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(result => 
        result.title.toLowerCase().includes(term) ||
        (result.abstract && result.abstract.toLowerCase().includes(term)) ||
        (result.authors && result.authors.some(author => 
          author && author.toLowerCase().includes(term)
        ))
      );
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let aValue = a[sortBy];
      let bValue = b[sortBy];
      
      // Handle special cases
      if (sortBy === 'title' || sortBy === 'abstract') {
        aValue = (aValue || '').toLowerCase();
        bValue = (bValue || '').toLowerCase();
      } else if (sortBy === 'authors') {
        aValue = (aValue && aValue.length > 0) ? aValue[0].toLowerCase() : '';
        bValue = (bValue && bValue.length > 0) ? bValue[0].toLowerCase() : '';
      }
      
      // Handle nulls/undefined
      if (aValue === undefined || aValue === null) return 1;
      if (bValue === undefined || bValue === null) return -1;
      
      // Compare based on direction
      if (sortOrder === 'asc') {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
      }
    });
    
    return filtered;
  };

  // Format source name for display - this is a regular function, not a hook
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

  // Get source color for chips - this is a regular function, not a hook
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

  // Toggle sort direction or change sort field
  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
  };

  // Render sort icon
  const renderSortIcon = (field) => {
    if (sortBy !== field) return null;
    
    return (
      <SortIcon 
        fontSize="small" 
        sx={{ 
          verticalAlign: 'middle', 
          transform: sortOrder === 'desc' ? 'rotate(180deg)' : 'none'
        }} 
      />
    );
  };

  // Handle tab changes
  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  // NOW we can do the conditional return after all hooks are called
  if (!results || results.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="body1">No results available</Typography>
      </Box>
    );
  }

  // Render a single source's results table
  const renderSourceTable = (source, sourceResults) => {
    const filteredResults = getFilteredAndSortedResults(sourceResults);
    
    return (
      <Box sx={{ mt: 2 }}>
        <Typography variant="subtitle1" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
          <Chip 
            label={formatSourceName(source)} 
            color={getSourceColor(source)} 
            sx={{ mr: 1 }} 
          />
          <span>{filteredResults.length} results</span>
        </Typography>
        
        <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                {columns.map((column) => (
                  <TableCell
                    key={column.id}
                    style={{ minWidth: column.minWidth }}
                    onClick={() => handleSort(column.id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    {column.label} {renderSortIcon(column.id)}
                  </TableCell>
                ))}
                <TableCell>Link</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredResults.map((row) => (
                <TableRow hover tabIndex={-1} key={`${source}-${row.title}`}>
                  <TableCell>{row.rank}</TableCell>
                  <TableCell>{row.title}</TableCell>
                  <TableCell>
                    {Array.isArray(row.authors) 
                      ? row.authors.slice(0, 3).join(', ') + (row.authors.length > 3 ? ', et al.' : '') 
                      : row.authors}
                  </TableCell>
                  <TableCell>{row.year}</TableCell>
                  <TableCell>
                    {row.url && (
                      <IconButton 
                        size="small" 
                        href={row.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                      >
                        <LaunchIcon fontSize="small" />
                      </IconButton>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {filteredResults.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} align="center">
                    No matching results found.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };

  return (
    <Card>
      <CardHeader 
        title="Detailed Search Results" 
        action={
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              size="small"
              variant="outlined"
              placeholder="Search in results..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: <FilterListIcon sx={{ mr: 1, color: 'action.active' }} />
              }}
            />
          </Box>
        }
      />
      <CardContent>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={currentTab} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
            <Tab label="All Sources" />
            {sources.map((source) => (
              <Tab 
                key={source} 
                label={formatSourceName(source)}
                icon={<Chip 
                  size="small" 
                  label={resultsBySource[source].length} 
                  color={getSourceColor(source)} 
                />} 
                iconPosition="end"
              />
            ))}
          </Tabs>
        </Box>
        
        {currentTab === 0 ? (
          // "All Sources" tab - side by side view
          <Grid container spacing={3}>
            {sources.map(source => (
              <Grid item xs={12} key={source}>
                {renderSourceTable(source, resultsBySource[source])}
              </Grid>
            ))}
          </Grid>
        ) : (
          // Individual source tab
          renderSourceTable(
            sources[currentTab - 1], 
            resultsBySource[sources[currentTab - 1]]
          )
        )}
      </CardContent>
    </Card>
  );
};

export default ResultsTable;